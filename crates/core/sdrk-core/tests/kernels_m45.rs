//! Tests of the M45 kernel `channelize` of `sdrk-core`: the construction rules, timing, the
//! consumption rule and `max_input_for_output`, the definition (an f64 sum written here,
//! independent of the polyphase form) for every channel count, `Fallback` against `Reference`,
//! block-size invariance with 20 chunk plans (seed 662, also with limited output space), the
//! history length, `reset`, the extreme inputs (silence, full scale, NaN, infinities, huge values)
//! and the thread-safety of the public types. The comparison with `Reference` runs only with the
//! feature `reference`.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::many_single_char_names
)]

use sdrk_core::kernels::channelize::{Channelize, ChannelizeState, history_len, taps_len};
use sdrk_core::kernels::fft::FftPlan;
use sdrk_core::{C32, Delay, Error, Fallback, Progress, Ratio, Timing};
use sdrk_device::{ErrorKind, SdrkError};

/// The seed of the chunk plans (M45: chunk invariance of the kernel); plan `k` uses
/// `SEED_PLANS * 100 + k`.
const SEED_PLANS: u64 = 662;
/// The seed of the signals and taps of these tests.
const SEED: u64 = 660;

type State = ChannelizeState<C32, Vec<f32>, Vec<C32>, Vec<C32>>;

/// Deterministic generator (xorshift64*), uniform in [-1, 1).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (v as f64 / (1u64 << 52) as f64 - 1.0) as f32
    }

    fn below(&mut self, n: usize) -> usize {
        ((f32::midpoint(self.next(), 1.0) * n as f32) as usize).min(n - 1)
    }
}

fn noise(rng: &mut Rng, n: usize, amp: f32) -> Vec<C32> {
    (0..n)
        .map(|_| C32::new(amp * rng.next(), amp * rng.next()))
        .collect()
}

/// Random taps of `m l` values with a sum of about 1.
fn random_taps(rng: &mut Rng, m: usize, l: usize) -> Vec<f32> {
    (0..m * l)
        .map(|_| (1.0 + rng.next()) / (m * l) as f32)
        .collect()
}

/// The tap storage of a state (`2 M L` values) with the prototype `taps` in its first half.
fn storage(taps: &[f32]) -> Vec<f32> {
    let mut t = vec![0.0f32; 2 * taps.len()];
    t[..taps.len()].copy_from_slice(taps);
    t
}

fn state(taps: &[f32], m: usize) -> State {
    let l = taps.len() / m;
    ChannelizeState::new(
        storage(taps),
        m,
        vec![C32::default(); history_len(m, l)],
        vec![C32::default(); m / 2],
    )
    .unwrap()
}

/// The whole input in one call with room for every step.
fn run_whole<K: Channelize<C32>>(k: &K, st: &mut State, input: &[C32]) -> Vec<C32> {
    let m = st.channels();
    let steps = (st.phase() + input.len()) / (m / 2);
    let mut out = vec![C32::default(); steps * m];
    let p = k.channelize(st, input, &mut out).unwrap();
    assert_eq!(p.consumed, input.len());
    assert_eq!(p.produced, steps * m);
    out
}

/// The definition `y_k[m] = sum_r h[r] x[t - r] e^(-j 2 pi k (t - r) / M)` in f64,
/// `t = (m + 1) M / 2 - 1`.
fn definition(h: &[f32], input: &[C32], m: usize) -> Vec<(f64, f64)> {
    let d = m / 2;
    let steps = input.len() / d;
    let mut out = Vec::with_capacity(steps * m);
    for step in 0..steps {
        let t = (step + 1) * d - 1;
        for k in 0..m {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (r, &hr) in h.iter().enumerate() {
                if r > t {
                    break;
                }
                let n = t - r;
                let x = input[n];
                let q = (k * n) % m;
                let theta = 2.0 * core::f64::consts::PI * q as f64 / m as f64;
                let (c, s) = (theta.cos(), -theta.sin());
                let (a, b) = (
                    f64::from(hr) * f64::from(x.re),
                    f64::from(hr) * f64::from(x.im),
                );
                re += a * c - b * s;
                im += a * s + b * c;
            }
            out.push((re, im));
        }
    }
    out
}

fn snr_db(want: &[(f64, f64)], got: &[C32]) -> f64 {
    assert_eq!(want.len(), got.len());
    let (mut p, mut e) = (0.0f64, 0.0f64);
    for (&(re, im), g) in want.iter().zip(got) {
        p += re * re + im * im;
        let (dr, di) = (re - f64::from(g.re), im - f64::from(g.im));
        e += dr * dr + di * di;
    }
    if e == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (p / e).log10()
    }
}

fn bits(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

/// The channel counts and taps per branch the tests run over.
const CONFIGS: [(usize, usize); 8] = [
    (4, 4),
    (4, 32),
    (8, 5),
    (16, 8),
    (32, 7),
    (64, 8),
    (128, 4),
    (256, 4),
];

#[test]
fn construction_rules() {
    // `taps` counts the prototype; the storage holds every tap twice.
    let ok = |taps: usize, m: usize, hist: usize, tw: usize| {
        ChannelizeState::<C32, _, _, _>::new(
            vec![0.0f32; 2 * taps],
            m,
            vec![C32::default(); hist],
            vec![C32::default(); tw],
        )
    };
    for m in [4usize, 8, 16, 32, 64, 128, 256] {
        for l in [4usize, 32] {
            let st = ok(m * l, m, m * l, m / 2).unwrap();
            assert_eq!(
                (st.channels(), st.taps_per_branch(), st.hop()),
                (m, l, m / 2)
            );
            // Only 16 channels and more hand out a plan the FFT kernel accepts (16 to 65536
            // points); 4 and 8 channels keep their twiddles to the radix-2 path.
            assert_eq!(st.plan().map(FftPlan::len), (m >= 16).then_some(m));
            assert_eq!(st.twiddles().len(), m / 2);
            assert_eq!((st.phase(), st.odd_step()), (0, false));
        }
    }
    let invalid = |r: Result<State, Error>| {
        let e = r.unwrap_err();
        assert!(matches!(e, Error::InvalidParameter(_)), "{e:?}");
        assert_eq!(e.kind(), ErrorKind::InvalidArgument);
    };
    // Channel counts: not a power of two, below 4, above 256.
    for m in [0usize, 1, 2, 3, 12, 24, 512] {
        invalid(ok(m.max(1) * 8, m, m.max(1) * 8, m / 2));
    }
    // Taps per branch 3 and 33, a tap count that is not a multiple of M.
    invalid(ok(16 * 3, 16, 16 * 3, 8));
    invalid(ok(16 * 33, 16, 16 * 33, 8));
    invalid(ok(16 * 8 + 1, 16, 16 * 9, 8));
    invalid(ok(0, 16, 128, 8));
    // A tap storage of odd length (one value short of 2 M L) is no layout of the state.
    invalid(ChannelizeState::<C32, _, _, _>::new(
        vec![0.0f32; taps_len(16, 8) - 1],
        16,
        vec![C32::default(); 128],
        vec![C32::default(); 8],
    ));
    // History shorter than M L, twiddles other than M / 2.
    assert_eq!(
        ok(128, 16, 127, 8).unwrap_err(),
        Error::LengthMismatch {
            expected: 128,
            got: 127
        }
    );
    assert_eq!(
        ok(128, 16, 128, 9).unwrap_err(),
        Error::LengthMismatch {
            expected: 8,
            got: 9
        }
    );
    assert_eq!(
        ok(16, 4, 16, 1).unwrap_err(),
        Error::LengthMismatch {
            expected: 2,
            got: 1
        }
    );
    assert_eq!(history_len(16, 8), 256);
    assert_eq!(history_len(usize::MAX, 2), usize::MAX);
    assert_eq!(taps_len(16, 8), 256);
    assert_eq!(taps_len(usize::MAX, 2), usize::MAX);
}

#[test]
fn the_taps_are_stored_time_reversed_twice_each_and_the_twiddles_filled() {
    let taps: Vec<f32> = (0..32).map(|i| i as f32).collect();
    for m in [4usize, 8] {
        let st = state(&taps, m);
        let want: Vec<f32> = taps.iter().rev().flat_map(|&t| [t, t]).collect();
        assert_eq!(st.taps(), &want[..]);
        assert!(st.plan().is_none());
        let tw = st.twiddles();
        assert_eq!(tw.len(), m / 2);
        for (k, w) in tw.iter().enumerate() {
            let theta = 2.0 * core::f64::consts::PI * k as f64 / m as f64;
            assert!((f64::from(w.re) - theta.cos()).abs() < 1e-7);
            assert!((f64::from(w.im) + theta.sin()).abs() < 1e-7);
        }
    }
}

#[test]
fn timing_is_two_steps_per_m_samples_and_half_the_prototype() {
    for (m, l) in CONFIGS {
        let st = state(&vec![0.0; m * l], m);
        assert_eq!(st.ratio(), Ratio::new(2, m as u64).unwrap());
        assert_eq!(st.group_delay(), Delay::new((m * l - 1) as u64, 2).unwrap());
    }
}

#[test]
fn consumption_follows_the_output_space() {
    let mut rng = Rng::new(SEED);
    let taps = random_taps(&mut rng, 16, 4);
    let input = noise(&mut rng, 100, 0.5);
    let mut st = state(&taps, 16);
    // No output space: the first 7 samples of the hop are taken, the 8th would complete it.
    assert_eq!(st.max_input_for_output(0), 7);
    let p = Fallback.channelize(&mut st, &input, &mut []).unwrap();
    assert_eq!(
        p,
        Progress {
            consumed: 7,
            produced: 0
        }
    );
    assert_eq!(st.phase(), 7);
    // Again without space: nothing more.
    let p = Fallback.channelize(&mut st, &input[7..], &mut []).unwrap();
    assert_eq!(
        p,
        Progress {
            consumed: 0,
            produced: 0
        }
    );
    // Two steps of space: two hops complete (1 + 8 samples), then 7 of the third.
    assert_eq!(st.max_input_for_output(2), 16);
    let mut out = vec![C32::default(); 32];
    let p = Fallback.channelize(&mut st, &input[7..], &mut out).unwrap();
    assert_eq!(
        p,
        Progress {
            consumed: 16,
            produced: 32
        }
    );
    assert_eq!((st.phase(), st.odd_step()), (7, false));
    // Short input: all of it, no output.
    let p = Fallback
        .channelize(&mut st, &input[23..23], &mut out)
        .unwrap();
    assert_eq!(p, Progress::default());
    // An output that is not a multiple of M.
    let e = Fallback
        .channelize(&mut st, &input, &mut [C32::default(); 20])
        .unwrap_err();
    assert_eq!(
        e,
        Error::LengthMismatch {
            expected: 16,
            got: 20
        }
    );
    assert_eq!(st.phase(), 7);
}

#[test]
fn the_fallback_follows_the_definition_for_every_channel_count() {
    let mut rng = Rng::new(SEED + 1);
    for (m, l) in CONFIGS {
        let taps = random_taps(&mut rng, m, l);
        let n = (m * l * 3).max(2048);
        let input = noise(&mut rng, n, 0.7);
        let want = definition(&taps, &input, m);
        let got = run_whole(&Fallback, &mut state(&taps, m), &input);
        let snr = snr_db(&want, &got);
        assert!(snr >= 120.0, "M {m}, L {l}: {snr:.1} dB");
    }
}

/// One tone on channel `k` (exactly at `k fs / M`) gives the DC gain of the prototype on that
/// channel and the gain at `(j - k) fs / M` on channel `j`, with the phase of the mixer at the
/// sample of every step (`e^(j 2 pi k t / M)` times the response).
#[test]
fn a_tone_on_a_channel_centre_appears_on_that_channel() {
    for m in [4usize, 8, 16, 64] {
        let l = 8;
        // A plain average: DC gain 1, zeros at every multiple of fs / (M L) except 0.
        let taps = vec![1.0 / (m * l) as f32; m * l];
        for k in [0usize, 1, m / 2, m - 1] {
            let input: Vec<C32> = (0..m * l * 4)
                .map(|n| {
                    let theta = 2.0 * core::f64::consts::PI * ((k * n) % m) as f64 / m as f64;
                    C32::new(theta.cos() as f32, theta.sin() as f32)
                })
                .collect();
            let out = run_whole(&Fallback, &mut state(&taps, m), &input);
            let last = &out[out.len() - m..];
            for (j, y) in last.iter().enumerate() {
                let p = f64::from(y.norm_sqr());
                if j == k {
                    assert!((p - 1.0).abs() < 1e-5, "M {m} k {k}: {p}");
                } else {
                    assert!(p < 1e-10, "M {m} k {k} j {j}: {p}");
                }
            }
        }
    }
}

#[cfg(feature = "reference")]
#[test]
fn fallback_and_reference_agree_within_the_contract() {
    use sdrk_core::Reference;
    let mut rng = Rng::new(SEED + 2);
    for (m, l) in CONFIGS {
        let taps = random_taps(&mut rng, m, l);
        let input = noise(&mut rng, 2 * m * l + 3 * m, 0.7);
        let want = definition(&taps, &input, m);
        let r = run_whole(&Reference, &mut state(&taps, m), &input);
        let f = run_whole(&Fallback, &mut state(&taps, m), &input);
        // The reference is the f64 model rounded once.
        assert!(snr_db(&want, &r) >= 140.0, "M {m}: {}", snr_db(&want, &r));
        let rr: Vec<(f64, f64)> = r
            .iter()
            .map(|c| (f64::from(c.re), f64::from(c.im)))
            .collect();
        let snr = snr_db(&rr, &f);
        assert!(snr >= 100.0, "M {m}, L {l}: {snr:.1} dB");
    }
}

#[cfg(feature = "reference")]
#[test]
fn the_reference_keeps_the_hop_boundaries_in_chunks() {
    use sdrk_core::Reference;
    let mut rng = Rng::new(SEED + 3);
    let taps = random_taps(&mut rng, 8, 4);
    let input = noise(&mut rng, 300, 0.5);
    let whole = run_whole(&Reference, &mut state(&taps, 8), &input);
    let mut st = state(&taps, 8);
    let mut got = Vec::new();
    for chunk in input.chunks(7) {
        let mut out = vec![C32::default(); 16];
        let p = Reference.channelize(&mut st, chunk, &mut out).unwrap();
        assert_eq!(p.consumed, chunk.len());
        got.extend_from_slice(&out[..p.produced]);
    }
    assert_eq!(bits(&whole), bits(&got));
}

/// Runs `input` in the chunks of plan `k`: chunk sizes from 1 to 300, and an output space of
/// 0 to 4 steps per call (the rest of a chunk is resubmitted).
fn run_chunked(st: &mut State, input: &[C32], k: u64) -> Vec<C32> {
    let m = st.channels();
    let mut rng = Rng::new(SEED_PLANS * 100 + k);
    let mut got = Vec::new();
    let mut pos = 0;
    let mut stalls = 0;
    while pos < input.len() {
        let size = (1 + rng.below(300)).min(input.len() - pos);
        let room = rng.below(5);
        let mut out = vec![C32::default(); room * m];
        let max = st.max_input_for_output(room);
        let p = Fallback
            .channelize(st, &input[pos..pos + size], &mut out)
            .unwrap();
        assert_eq!(p.consumed, size.min(max));
        got.extend_from_slice(&out[..p.produced]);
        pos += p.consumed;
        stalls = if p.consumed == 0 { stalls + 1 } else { 0 };
        assert!(stalls < 100, "no progress");
    }
    got
}

#[test]
fn block_size_invariance_with_20_chunk_plans() {
    let mut rng = Rng::new(SEED + 4);
    for (j, (m, l)) in [(4usize, 4usize), (16, 8), (64, 8), (256, 4)]
        .into_iter()
        .enumerate()
    {
        let taps = random_taps(&mut rng, m, l);
        let input = noise(&mut rng, 6000, 0.7);
        let mut whole_state = state(&taps, m);
        let whole = run_whole(&Fallback, &mut whole_state, &input);
        for k in 0..20 {
            let mut st = state(&taps, m);
            let got = run_chunked(&mut st, &input, 100 * j as u64 + k);
            assert_eq!(bits(&whole), bits(&got), "M {m}, plan {k}");
            assert_eq!(
                (st.phase(), st.odd_step()),
                (whole_state.phase(), whole_state.odd_step())
            );
        }
    }
}

#[test]
fn the_output_does_not_depend_on_the_history_length() {
    let mut rng = Rng::new(SEED + 5);
    for (m, l) in [(4usize, 4usize), (16, 8), (32, 5)] {
        let taps = random_taps(&mut rng, m, l);
        let input = noise(&mut rng, 3000, 0.7);
        let want = run_whole(&Fallback, &mut state(&taps, m), &input);
        for hist in [m * l, m * l + 1, m * l + m / 2, 3 * m * l + 5] {
            let mut st = ChannelizeState::new(
                storage(&taps),
                m,
                vec![C32::default(); hist],
                vec![C32::default(); m / 2],
            )
            .unwrap();
            assert_eq!(
                bits(&run_whole(&Fallback, &mut st, &input)),
                bits(&want),
                "{hist}"
            );
        }
    }
}

#[test]
fn reset_restarts_at_hop_zero() {
    let mut rng = Rng::new(SEED + 6);
    let taps = random_taps(&mut rng, 16, 8);
    let input = noise(&mut rng, 1000, 0.7);
    let fresh = run_whole(&Fallback, &mut state(&taps, 16), &input);
    let mut st = state(&taps, 16);
    run_whole(&Fallback, &mut st, &input[..333]);
    assert_eq!(st.phase(), 333 % 8);
    st.reset();
    assert_eq!((st.phase(), st.odd_step()), (0, false));
    assert!(st.history().iter().all(|v| *v == C32::default()));
    assert_eq!(bits(&run_whole(&Fallback, &mut st, &input)), bits(&fresh));
}

#[test]
fn silence_gives_exact_zeros() {
    for (m, l) in CONFIGS {
        let mut rng = Rng::new(SEED + 7);
        let taps = random_taps(&mut rng, m, l);
        let out = run_whole(
            &Fallback,
            &mut state(&taps, m),
            &vec![C32::default(); 4 * m * l],
        );
        assert!(out.iter().all(|v| v.re == 0.0 && v.im == 0.0), "M {m}");
    }
}

#[test]
fn full_scale_stays_finite_and_bounded() {
    for (m, l) in CONFIGS {
        let mut rng = Rng::new(SEED + 8);
        let taps = random_taps(&mut rng, m, l);
        let gain: f32 = taps.iter().map(|t| t.abs()).sum();
        // Alternating +-(1 - 2^-15) on both components: the bound of every output is the sum
        // of the tap magnitudes times the largest input magnitude.
        let a = 1.0 - 1.0 / 32768.0;
        let input: Vec<C32> = (0..4 * m * l)
            .map(|n| {
                let s = if n % 2 == 0 { a } else { -a };
                C32::new(s, -s)
            })
            .collect();
        let out = run_whole(&Fallback, &mut state(&taps, m), &input);
        let bound = gain * core::f32::consts::SQRT_2 * 1.0001;
        assert!(out.iter().all(|v| v.norm_sqr().sqrt() <= bound), "M {m}");
    }
}

/// Indices of the output steps whose window holds input sample `n`.
fn steps_reading(n: usize, m: usize, l: usize, steps: usize) -> core::ops::Range<usize> {
    let d = m / 2;
    // Step `s` reads `t - N + 1 ..= t` with `t = (s + 1) D - 1`.
    let first = (n + 1).div_ceil(d).saturating_sub(1);
    let last = ((n + m * l) / d).min(steps);
    first..last
}

#[test]
fn non_finite_input_spoils_only_the_steps_that_read_it() {
    let specials = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX];
    for (m, l) in [(4usize, 4usize), (8, 5), (16, 8), (64, 4)] {
        for (i, &v) in specials.iter().enumerate() {
            let mut rng = Rng::new(SEED + 9);
            let taps = random_taps(&mut rng, m, l);
            let mut input = noise(&mut rng, 6 * m * l, 0.5);
            let at = 2 * m * l + 3 + i;
            input[at] = C32::new(v, if i % 2 == 0 { 0.25 } else { v });
            let out = run_whole(&Fallback, &mut state(&taps, m), &input);
            let steps = out.len() / m;
            let spoiled = steps_reading(at, m, l, steps);
            for (s, group) in out.chunks_exact(m).enumerate() {
                let finite = group.iter().all(|c| c.re.is_finite() && c.im.is_finite());
                if !spoiled.contains(&s) {
                    assert!(finite, "M {m}, value {v}: step {s} outside {spoiled:?}");
                } else if v.is_nan() || v.is_infinite() {
                    assert!(!finite, "M {m}, value {v}: step {s} inside {spoiled:?}");
                }
            }
            // The steps after the window are exactly those of the input without the value.
            input[at] = C32::default();
            let clean = run_whole(&Fallback, &mut state(&taps, m), &input);
            assert_eq!(
                bits(&out[spoiled.end * m..]),
                bits(&clean[spoiled.end * m..]),
                "M {m}"
            );
        }
    }
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<State>();
    assert_send_sync::<ChannelizeState<C32, [f32; 128], [C32; 128], [C32; 8]>>();
    assert_send_sync::<ChannelizeState<C32, &'static mut [f32], &'static mut [C32], &'static [C32]>>(
    );
}

/// The 4- and 8-point twiddles of a small bank never form a plan the FFT kernel would run: the
/// state hands out no plan below 16 channels, and the kernel's own constructor rejects the size.
#[test]
fn a_small_bank_hands_out_no_fft_plan() {
    use sdrk_core::kernels::fft::{Direction, Fft, Scaling};
    let taps = vec![0.5f32; 64];
    for m in [4usize, 8] {
        let st = state(&taps[..m * 8], m);
        assert!(st.plan().is_none());
        assert_eq!(st.twiddles().len(), m / 2);
        let e = FftPlan::<C32, _>::new(m, vec![C32::default(); m / 2]).unwrap_err();
        assert!(matches!(e, Error::InvalidParameter(_)), "{e:?}");
    }
    let st = state(&taps[..16 * 4], 16);
    let plan = st.plan().expect("16 channels use the FFT kernel");
    let mut data = vec![C32::new(1.0, 0.0); 16];
    Fallback
        .fft(plan, &mut data, Direction::Forward, Scaling::Default)
        .unwrap();
    assert!((data[0].re - 16.0).abs() < 1e-5 && data[1].re.abs() < 1e-5);
}
