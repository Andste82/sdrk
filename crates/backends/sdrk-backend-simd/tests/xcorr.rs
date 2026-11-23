//! `xcorr_fft` (M50 test 8) at the level of the running build (CI runs every level).
//!
//! - The spectral-product helper against the scalar formula of `sdrk_core::kernels::xcorr`
//!   (`spectral_product_one`): every lane by `to_bits()`, a NaN result by class (D296: `∞ · 0`
//!   and other NaN-producing products may differ in NaN sign and payload), on every length from
//!   0 to 67 and 4096, at every start offset within a vector, on random values and on the special
//!   values (`±0`, `±∞`, NaN, `f32::MAX`, subnormals).
//! - The kernel against `Reference` and `Fallback`: at least 100 dB (the contract floor) for the
//!   `L`, `N` of the golden cases and at `N` 16 and 65536 (at `N` 65536 with `L` 32768 against
//!   the fallback only: the f64 reference of a whole block is `N L` complex products); the
//!   history after every call equal to the fallback's bit for bit (a copy of input samples).
//! - Chunk invariance in 20 chunk plans (seed 705, an output room of 0 to 3 blocks per call):
//!   the output and the history after every call equal to the whole run bit for bit.
//! - The extremes: silence gives exact zeros; NaN or infinity in one sample spoils at most the
//!   two blocks whose windows hold it, every later block equals the clean run bit for bit.
//! - The golden cases of M50 §4 (`tests/golden/vectors/xcorr_fft`): the SNR floor and the
//!   regression guard of each case.
//!
//! `backend_for("xcorr_fft", ..)` is checked by the facade (`sdrk/tests/facade.rs`), which knows
//! the backends. Signals, taps and special values: seed 708.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::print_stderr
)]

mod common;

use std::path::PathBuf;

use common::{Rng, c32_noise, snr_db};
use sdrk_backend_simd::Simd;
use sdrk_backend_simd::xcorr::spectral_product;
use sdrk_core::kernels::fft::FftPlan;
use sdrk_core::kernels::xcorr::{XcorrFft, XcorrFftState, spectral_product_one};
use sdrk_core::{C32, Error, Fallback, Reference};

/// The seed of the signals, taps and special values of these tests.
const SEED: u64 = 708;
/// The seed of the chunk plans (M50 test 4); plan `k` uses `SEED_PLANS * 100 + k`.
const SEED_PLANS: u64 = 705;
/// The contract floor in dB.
const FLOOR_DB: f64 = 100.0;

type State = XcorrFftState<C32, Vec<C32>, Vec<C32>, Vec<C32>, Vec<C32>>;

fn z(n: usize) -> Vec<C32> {
    vec![C32::default(); n]
}

fn state(taps: &[C32], n: usize) -> State {
    let plan = FftPlan::new(n, z(n / 2)).unwrap();
    XcorrFftState::new(taps, z(n), z(taps.len() - 1), z(n), plan).unwrap()
}

fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn bits(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
}

/// `C32` special values: signed zeros, infinities, subnormals, NaN payloads of both signs, the
/// extremes of the finite range and ordinary values.
fn specials() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x007F_FFFF),
        f32::MIN_POSITIVE,
        f32::from_bits(0x7FC0_0000),
        f32::from_bits(0xFFC0_0001),
        f32::from_bits(0x7F80_0001),
        f32::MAX,
        f32::MIN,
        1.0,
        -1.0,
        0.5,
        -0.75,
        3.0e19,
        -2.0e-20,
    ]
}

/// `n` samples: random in the unit square, or (with `special`) every part drawn from
/// [`specials`].
fn samples(rng: &mut Rng, n: usize, special: bool) -> Vec<C32> {
    let sp = specials();
    (0..n)
        .map(|_| {
            if special {
                C32::new(sp[rng.below(sp.len())], sp[rng.below(sp.len())])
            } else {
                C32::new(rng.unit(), rng.unit())
            }
        })
        .collect()
}

/// The helper on `w[offset..offset + n]`, `h[offset..offset + n]` against the scalar formula.
fn check_product(w: &[C32], h: &[C32], offset: usize, n: usize, what: &str) {
    let mut got = w.to_vec();
    spectral_product(&mut got[offset..offset + n], &h[offset..offset + n]).unwrap();
    for (k, (g, (a, b))) in got.iter().zip(w.iter().zip(h)).enumerate() {
        let want = if (offset..offset + n).contains(&k) {
            spectral_product_one(*a, *b)
        } else {
            *a
        };
        assert!(
            same_f32(g.re, want.re) && same_f32(g.im, want.im),
            "{what} n {n} offset {offset} bin {k}: {a:?} * {b:?} = {g:?}, want {want:?}"
        );
    }
}

#[test]
fn the_spectral_product_has_the_bits_of_the_scalar_formula() {
    let mut rng = Rng::new(SEED);
    let lengths: Vec<usize> = (0..=67).chain([4096]).collect();
    for &n in &lengths {
        for special in [false, true] {
            let w = samples(&mut rng, n + 3, special);
            let h = samples(&mut rng, n + 3, special);
            for offset in 0..=3 {
                if offset + n <= w.len() {
                    check_product(
                        &w,
                        &h,
                        offset,
                        n,
                        if special { "special" } else { "random" },
                    );
                }
            }
        }
    }
    // Every pair of special values, in every lane position.
    let sp = specials();
    let values: Vec<C32> = sp
        .iter()
        .flat_map(|&r| sp.iter().map(move |&i| C32::new(r, i)))
        .collect();
    for shift in 0..16 {
        let w: Vec<C32> = values
            .iter()
            .cycle()
            .skip(shift)
            .take(values.len())
            .copied()
            .collect();
        for h in values.iter().step_by(7) {
            let hv = vec![*h; w.len()];
            check_product(&w, &hv, 0, w.len(), "pairs");
        }
    }
}

#[test]
fn the_spectral_product_rejects_unequal_lengths() {
    let mut w = z(5);
    assert!(matches!(
        spectral_product(&mut w, &z(4)),
        Err(Error::LengthMismatch { .. })
    ));
}

/// The whole input in one call with room for every block.
fn run_whole<K: XcorrFft<C32>>(k: &K, st: &mut State, input: &[C32]) -> Vec<C32> {
    let mut out = z(input.len());
    let p = k.xcorr_fft(st, input, &mut out).unwrap();
    let b = st.block_size();
    assert_eq!(p.consumed, input.len() / b * b);
    assert_eq!(p.produced, p.consumed);
    out.truncate(p.produced);
    out
}

#[test]
fn the_kernel_reaches_the_floor_against_the_reference_and_the_fallback() {
    let mut rng = Rng::new(SEED + 1);
    // (L, N, blocks): the golden sizes, the smallest N, the largest N.
    for (l, n, blocks) in [
        (127usize, 512usize, 4usize),
        (1023, 4096, 3),
        (65, 256, 4),
        (300, 1024, 4),
        (1, 16, 5),
        (8, 16, 7),
        (5, 16, 6),
        (300, 65_536, 2),
    ] {
        let taps = c32_noise(&mut rng, l, 1.0 / (l as f32).sqrt());
        let input = c32_noise(&mut rng, blocks * (n - l + 1) + 7, 0.9);
        let mut simd_state = state(&taps, n);
        let mut fallback_state = state(&taps, n);
        let simd = run_whole(&Simd, &mut simd_state, &input);
        let fallback = run_whole(&Fallback, &mut fallback_state, &input);
        let reference = run_whole(&Reference, &mut state(&taps, n), &input);
        let vs_reference = snr_db(&reference, &simd);
        let vs_fallback = snr_db(&fallback, &simd);
        assert!(
            vs_reference >= FLOOR_DB,
            "L {l} N {n}: {vs_reference:.1} dB against the reference"
        );
        assert!(
            vs_fallback >= FLOOR_DB,
            "L {l} N {n}: {vs_fallback:.1} dB against the fallback"
        );
        assert_eq!(
            bits(simd_state.history()),
            bits(fallback_state.history()),
            "L {l} N {n}"
        );
    }
}

#[test]
fn the_largest_size_agrees_with_the_fallback() {
    let mut rng = Rng::new(SEED + 2);
    let (l, n) = (32_768usize, 65_536usize);
    let taps = c32_noise(&mut rng, l, 1.0 / (l as f32).sqrt());
    let input = c32_noise(&mut rng, 2 * (n - l + 1), 0.9);
    let mut simd_state = state(&taps, n);
    let mut fallback_state = state(&taps, n);
    let simd = run_whole(&Simd, &mut simd_state, &input);
    let fallback = run_whole(&Fallback, &mut fallback_state, &input);
    let db = snr_db(&fallback, &simd);
    assert!(db >= FLOOR_DB, "{db:.1} dB against the fallback");
    assert_eq!(bits(simd_state.history()), bits(fallback_state.history()));
}

/// The input in the chunks of plan `k` with an output room of 0 to 3 blocks per call, through
/// `Simd` and `Fallback` side by side: the outputs and, after every call, the histories of both
/// (equal bit for bit) and of `Simd` (returned for the comparison with the whole run).
fn run_chunked(taps: &[C32], n: usize, input: &[C32], k: u64) -> (Vec<C32>, Vec<Vec<C32>>) {
    let mut s = state(taps, n);
    let mut f = state(taps, n);
    let b = s.block_size();
    let mut plan = Rng::new(SEED_PLANS * 100 + k);
    let mut out = Vec::new();
    let mut histories = Vec::new();
    let mut pending: Vec<C32> = Vec::new();
    let mut pos = 0;
    while pos < input.len() || pending.len() >= b {
        let take = (1 + plan.below(3 * b)).min(input.len() - pos);
        pending.extend_from_slice(&input[pos..pos + take]);
        pos += take;
        let room = plan.below(4) * b + plan.below(b);
        let (mut ys, mut yf) = (z(room), z(room));
        let ps = Simd.xcorr_fft(&mut s, &pending, &mut ys).unwrap();
        let pf = Fallback.xcorr_fft(&mut f, &pending, &mut yf).unwrap();
        assert_eq!(ps, pf, "plan {k}");
        assert_eq!(
            ps.consumed,
            (pending.len() / b).min(room / b) * b,
            "plan {k}"
        );
        assert_eq!(bits(s.history()), bits(f.history()), "plan {k}");
        out.extend_from_slice(&ys[..ps.produced]);
        histories.push(s.history().to_vec());
        pending.drain(..ps.consumed);
        if pos == input.len() && ps.consumed == 0 && pending.len() < b {
            break;
        }
    }
    (out, histories)
}

#[test]
fn the_output_does_not_depend_on_the_chunking() {
    let mut rng = Rng::new(SEED + 3);
    for (l, n) in [(127usize, 512usize), (8, 16), (65, 256)] {
        let taps = c32_noise(&mut rng, l, 1.0 / (l as f32).sqrt());
        let b = n - l + 1;
        let input = c32_noise(&mut rng, 9 * b + 3, 0.8);
        let whole = run_whole(&Simd, &mut state(&taps, n), &input);
        // The history after every whole block of the whole run.
        let mut block_history = Vec::new();
        let mut st = state(&taps, n);
        for x in input.chunks_exact(b) {
            let mut y = z(b);
            Simd.xcorr_fft(&mut st, x, &mut y).unwrap();
            block_history.push(st.history().to_vec());
        }
        for k in 0..20 {
            let (chunked, histories) = run_chunked(&taps, n, &input, k);
            assert_eq!(bits(&chunked), bits(&whole), "L {l} N {n} plan {k}");
            for h in &histories {
                assert!(
                    h.iter().all(|v| *v == C32::default())
                        || block_history.iter().any(|w| bits(w) == bits(h)),
                    "L {l} N {n} plan {k}: a history that the whole run never had"
                );
            }
            assert_eq!(
                bits(histories.last().unwrap()),
                bits(block_history.last().unwrap()),
                "L {l} N {n} plan {k}"
            );
        }
    }
}

#[test]
fn silence_and_non_finite_samples() {
    let mut rng = Rng::new(SEED + 4);
    let (l, n) = (100usize, 256usize);
    let b = n - l + 1;
    let taps = c32_noise(&mut rng, l, 0.1);
    // Silence: exact zeros.
    let silence = run_whole(&Simd, &mut state(&taps, n), &z(5 * b));
    assert!(silence.iter().all(|v| v.re == 0.0 && v.im == 0.0));
    // One bad sample in block 1 near its end (also in the window of block 2 through the
    // history) or at its start (block 1 only).
    let clean = c32_noise(&mut rng, 6 * b, 0.9);
    let want = run_whole(&Simd, &mut state(&taps, n), &clean);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for (at, spoiled) in [(2 * b - 5, 2usize), (b, 1)] {
            let mut x = clean.clone();
            x[at].re = bad;
            let got = run_whole(&Simd, &mut state(&taps, n), &x);
            assert_eq!(bits(&got[..b]), bits(&want[..b]), "{bad} at {at}: block 0");
            for v in &got[b..(1 + spoiled) * b] {
                assert!(!v.re.is_finite() || !v.im.is_finite(), "{bad} at {at}");
            }
            let later = (1 + spoiled) * b;
            assert_eq!(bits(&got[later..]), bits(&want[later..]), "{bad} at {at}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The golden cases of M50 §4.

fn golden(file: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("tests")
        .join("golden")
        .join("vectors")
        .join("xcorr_fft")
        .join(file);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The value of the JSON field `"name": value` (numbers).
fn field(json: &str, name: &str) -> String {
    let key = format!("\"{name}\": ");
    let at = json.find(&key).unwrap_or_else(|| panic!("no field {name}")) + key.len();
    json[at..]
        .chars()
        .take_while(|c| !matches!(c, ',' | '\n' | '}'))
        .collect::<String>()
        .trim()
        .to_owned()
}

fn c32s(b: &[u8]) -> Vec<C32> {
    b.chunks_exact(8)
        .map(|c| {
            C32::new(
                f32::from_le_bytes(c[..4].try_into().unwrap()),
                f32::from_le_bytes(c[4..].try_into().unwrap()),
            )
        })
        .collect()
}

fn f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// SNR in dB of `got` against the exact pairs `want`.
fn snr_exact(want: &[f64], got: &[C32]) -> f64 {
    let (mut s, mut e) = (0.0, 0.0);
    for (w, g) in want.chunks_exact(2).zip(got) {
        s += w[0] * w[0] + w[1] * w[1];
        e += (w[0] - f64::from(g.re)).powi(2) + (w[1] - f64::from(g.im)).powi(2);
    }
    if e == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (s / e).log10()
    }
}

#[test]
fn the_golden_cases_hold_at_this_level() {
    for (name, seed) in [
        ("code_127_n512", 700u64),
        ("code_1023_n4096", 701),
        ("rrc_65_n256", 702),
        ("rand_300_n1024", 703),
    ] {
        let json = String::from_utf8(golden(&format!("{name}.json"))).unwrap();
        // The case seed is the top-level field after `params` (whose `seed` is the PRBS start
        // value of the code cases).
        let top = &json[json.rfind("\"provenance\"").unwrap()..];
        assert_eq!(field(top, "seed"), seed.to_string(), "{name}");
        let n: usize = field(&json, "fft_size").parse().unwrap();
        let min: f64 = field(&json, "snr_db_min").parse().unwrap();
        let guard: f64 = field(&json, "snr_db_guard").parse().unwrap();
        let taps = c32s(&golden(&format!("{name}.taps.bin")));
        let input = c32s(&golden(&format!("{name}.input.bin")));
        let expected = f64s(&golden(&format!("{name}.expected.bin")));
        let got = run_whole(&Simd, &mut state(&taps, n), &input);
        assert_eq!(2 * got.len(), expected.len(), "{name}");
        let db = snr_exact(&expected, &got);
        eprintln!("{name}: {db:.2} dB");
        assert!(db >= min, "{name}: {db:.2} dB below the floor {min} dB");
        assert!(db >= guard, "{name}: {db:.2} dB below the guard {guard} dB");
    }
}
