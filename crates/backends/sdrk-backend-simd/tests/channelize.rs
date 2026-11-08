//! `channelize` (M45) at the level of the running build against the fallback and against the
//! f64 model of `Reference`: within the 100 dB of the contract for every channel count from 4 to
//! 256 and the taps per branch 4, 8, 24 and 32; the hop boundaries in 20 chunk plans (seed 662)
//! with limited output space, where the output must be bit-identical with the whole run of the
//! same backend; history lengths around `M L + M / 2` (the shortest that lets the NEON path sum
//! two hops at once, I45-1), bit for bit; buffers at every start offset within a vector; `reset`;
//! the extreme inputs
//! (silence, full scale, NaN and infinities spoil only the steps that read them); and the error
//! for an output that is not a whole number of steps. Signals and taps use seed 672.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::float_cmp
)]

mod common;

use common::{Rng, c32_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::channelize::{Channelize, ChannelizeState, history_len};
use sdrk_core::{C32, Error, Fallback, Reference};

/// The seed of the signals and taps of these tests.
const SEED: u64 = 672;
/// The seed of the chunk plans (M45 test 1); plan `k` uses `SEED_PLANS * 100 + k`.
const SEED_PLANS: u64 = 662;
/// The floor of the numeric contract (`docs/kernels.md`).
const CONTRACT_DB: f64 = 100.0;

type State = ChannelizeState<C32, Vec<f32>, Vec<C32>, Vec<C32>>;

/// Random taps of `m l` values with a sum of about 1.
fn random_taps(rng: &mut Rng, m: usize, l: usize) -> Vec<f32> {
    (0..m * l)
        .map(|_| rng.unit() * 2.0 / (m * l) as f32)
        .collect()
}

fn state(taps: &[f32], m: usize) -> State {
    let mut stored = vec![0.0f32; 2 * taps.len()];
    stored[..taps.len()].copy_from_slice(taps);
    ChannelizeState::new(
        stored,
        m,
        vec![C32::default(); history_len(m, taps.len() / m)],
        vec![C32::default(); m / 2],
    )
    .unwrap()
}

/// The whole input in one call with room for every step.
fn run_whole<K: Channelize<C32>>(k: &K, st: &mut State, input: &[C32]) -> Vec<C32> {
    let m = st.channels();
    let mut out = vec![C32::default(); (input.len() / (m / 2) + 1) * m];
    let p = k.channelize(st, input, &mut out).unwrap();
    assert_eq!(p.consumed, input.len());
    out.truncate(p.produced);
    out
}

/// `10 log10(sum |want|^2 / sum |want - got|^2)`.
fn snr_db(want: &[C32], got: &[C32]) -> f64 {
    assert_eq!(want.len(), got.len());
    let (mut s, mut e) = (0.0f64, 0.0f64);
    for (w, g) in want.iter().zip(got) {
        let (wr, wi) = (f64::from(w.re), f64::from(w.im));
        s += wr * wr + wi * wi;
        e += (wr - f64::from(g.re)).powi(2) + (wi - f64::from(g.im)).powi(2);
    }
    if e == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (s / e).log10()
    }
}

fn bits(x: &[C32]) -> Vec<(u32, u32)> {
    let b = |v: f32| if v.is_nan() { u32::MAX } else { v.to_bits() };
    x.iter().map(|v| (b(v.re), b(v.im))).collect()
}

#[test]
fn simd_agrees_with_the_fallback_and_the_reference_for_every_size() {
    let mut rng = Rng::new(SEED);
    for m in [4, 8, 16, 32, 64, 128, 256] {
        for l in [4, 8, 24, 32] {
            let taps = random_taps(&mut rng, m, l);
            let input = c32_noise(&mut rng, (m / 2) * (3 * l + 7) + 3, 0.9);
            let simd = run_whole(&Simd, &mut state(&taps, m), &input);
            let fallback = run_whole(&Fallback, &mut state(&taps, m), &input);
            let reference = run_whole(&Reference, &mut state(&taps, m), &input);
            let vs_fallback = snr_db(&fallback, &simd);
            let vs_reference = snr_db(&reference, &simd);
            assert!(
                vs_fallback >= CONTRACT_DB,
                "m {m} l {l}: {vs_fallback:.1} dB against the fallback"
            );
            assert!(
                vs_reference >= CONTRACT_DB,
                "m {m} l {l}: {vs_reference:.1} dB against the reference"
            );
        }
    }
}

/// The input in the chunks of plan `k`, with output space for at most `room` steps per call.
fn run_chunked(st: &mut State, input: &[C32], k: u64) -> Vec<C32> {
    let m = st.channels();
    let mut plan = Rng::new(SEED_PLANS * 100 + k);
    let mut out = Vec::new();
    let mut rest = input;
    while !rest.is_empty() {
        let take = (1 + plan.below(3 * m)).min(rest.len());
        let room = plan.below(4);
        let mut buf = vec![C32::default(); room * m];
        let p = Simd.channelize(st, &rest[..take], &mut buf).unwrap();
        out.extend_from_slice(&buf[..p.produced]);
        rest = &rest[p.consumed..];
    }
    out
}

#[test]
fn the_hop_boundaries_do_not_depend_on_the_chunking() {
    let mut rng = Rng::new(SEED + 1);
    for m in [4, 16, 64] {
        let taps = random_taps(&mut rng, m, 8);
        let input = c32_noise(&mut rng, 40 * m + 5, 0.7);
        let whole = run_whole(&Simd, &mut state(&taps, m), &input);
        for k in 0..20 {
            let chunked = run_chunked(&mut state(&taps, m), &input, k);
            let n = chunked.len().min(whole.len());
            assert!(n + 2 * m >= whole.len(), "m {m} plan {k}: too few steps");
            assert_eq!(bits(&chunked[..n]), bits(&whole[..n]), "m {m} plan {k}");
        }
    }
}

#[test]
fn the_history_length_does_not_change_the_bits() {
    let mut rng = Rng::new(SEED + 6);
    for (m, l) in [(16usize, 4usize), (16, 8), (64, 5)] {
        let taps = random_taps(&mut rng, m, l);
        let input = c32_noise(&mut rng, 30 * m + 3, 0.7);
        let want = run_whole(&Simd, &mut state(&taps, m), &input);
        let n = m * l;
        for hist in [n, n + m / 2 - 1, n + m / 2, n + m / 2 + 1, 3 * n + 5] {
            let mut stored = vec![0.0f32; 2 * n];
            stored[..n].copy_from_slice(&taps);
            let mut st = ChannelizeState::new(
                stored,
                m,
                vec![C32::default(); hist],
                vec![C32::default(); m / 2],
            )
            .unwrap();
            assert_eq!(
                bits(&run_whole(&Simd, &mut st, &input)),
                bits(&want),
                "m {m} l {l} history {hist}"
            );
        }
    }
}

#[test]
fn the_start_offset_of_the_buffers_does_not_matter() {
    let mut rng = Rng::new(SEED + 2);
    let m = 32;
    let taps = random_taps(&mut rng, m, 8);
    let input = c32_noise(&mut rng, 20 * m + 3, 0.8);
    let want = run_whole(&Simd, &mut state(&taps, m), &input);
    for off in 0..4 {
        let mut src = vec![C32::default(); input.len() + 4];
        src[off..off + input.len()].copy_from_slice(&input);
        let mut dst = vec![C32::default(); want.len() + 4];
        let p = Simd
            .channelize(
                &mut state(&taps, m),
                &src[off..off + input.len()],
                &mut dst[off..off + want.len()],
            )
            .unwrap();
        assert_eq!(p.produced, want.len(), "offset {off}");
        assert_eq!(
            bits(&dst[off..off + want.len()]),
            bits(&want),
            "offset {off}"
        );
    }
}

#[test]
fn reset_restarts_at_hop_zero() {
    let mut rng = Rng::new(SEED + 3);
    let taps = random_taps(&mut rng, 16, 8);
    let input = c32_noise(&mut rng, 333, 0.5);
    let mut st = state(&taps, 16);
    let first = run_whole(&Simd, &mut st, &input);
    st.reset();
    let again = run_whole(&Simd, &mut st, &input);
    assert_eq!(bits(&first), bits(&again));
}

#[test]
fn silence_and_full_scale() {
    let mut rng = Rng::new(SEED + 4);
    let m = 16;
    let taps = random_taps(&mut rng, m, 8);
    let silent = run_whole(&Simd, &mut state(&taps, m), &vec![C32::default(); 400]);
    assert!(silent.iter().all(|v| v.re == 0.0 && v.im == 0.0));
    let full: Vec<C32> = (0..400)
        .map(|n| {
            let s = if n % 3 == 0 { -1.0 } else { 1.0 };
            C32::new(s, -s)
        })
        .collect();
    let simd = run_whole(&Simd, &mut state(&taps, m), &full);
    let fallback = run_whole(&Fallback, &mut state(&taps, m), &full);
    assert!(simd.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
    assert!(snr_db(&fallback, &simd) >= CONTRACT_DB);
}

#[test]
fn non_finite_input_spoils_only_the_steps_that_read_it() {
    let mut rng = Rng::new(SEED + 5);
    let (m, l) = (16, 4);
    let n = m * l;
    let taps: Vec<f32> = random_taps(&mut rng, m, l)
        .iter()
        .map(|t| t.abs() + 1e-3)
        .collect();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut input = c32_noise(&mut rng, 40 * m, 0.5);
        let at = 10 * m + 3;
        input[at] = C32::new(bad, 0.25);
        let out = run_whole(&Simd, &mut state(&taps, m), &input);
        for (step, values) in out.chunks_exact(m).enumerate() {
            // Step `j` reads the input from `(j + 1) D - N` to `(j + 1) D - 1`.
            let end = (step + 1) * (m / 2);
            let reads = end > at && end <= at + n;
            let finite = values.iter().all(|v| v.re.is_finite() && v.im.is_finite());
            assert_eq!(
                finite, !reads,
                "{bad}: step {step} (reads the sample: {reads})"
            );
        }
    }
}

#[test]
fn an_output_of_part_of_a_step_is_rejected() {
    let taps = vec![0.0f32; 64];
    let mut st = state(&taps, 16);
    let mut out = vec![C32::default(); 20];
    assert!(matches!(
        Simd.channelize(&mut st, &[C32::default(); 8], &mut out),
        Err(Error::LengthMismatch { .. })
    ));
}
