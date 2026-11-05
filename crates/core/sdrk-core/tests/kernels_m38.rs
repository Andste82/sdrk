//! Tests of the M38 kernel of `sdrk-core`: `symbol_sync`. The construction rules, timing, the
//! interpolation at `mu = 0` (the input samples themselves) and at `mu = 0.5` (exact on a cubic),
//! a constant input, lock on clean signals at integer `sps` (Gardner and Mueller–Müller, real
//! and complex), block-size invariance (also with a full output), `reset`, non-finite input and
//! the thread-safety of the public types. The comparison of `Fallback` with `Reference` runs
//! only with the feature `reference`.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::manual_midpoint
)]

use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncGains, SymbolSyncState, Ted};
use sdrk_core::{C32, CQ15, Delay, Error, Fallback, Progress, Q15, Ratio, RatioKind, Timing};
use sdrk_device::{ErrorKind, SdrkError};

const DAMP: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// Deterministic generator (xorshift64*), uniform in [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (v as f64 / (1u64 << 52) as f64 - 1.0) as f32
    }

    fn bit(&mut self) -> bool {
        self.next() >= 0.0
    }

    fn below(&mut self, n: usize) -> usize {
        (((self.next() + 1.0) * 0.5) * n as f32) as usize % n
    }
}

/// The value at time `x` (in samples) of symbols `a` with `sps` samples each, the first centred
/// at `offset`, joined by straight lines between the symbol centres.
fn linear(a: &[f32], sps: f64, offset: f64, x: f64) -> f64 {
    let u = (x - offset) / sps;
    let k = libm::floor(u);
    let frac = u - k;
    let get = |k: f64| -> f64 {
        if k < 0.0 {
            f64::from(a[0])
        } else {
            f64::from(*a.get(k as usize).unwrap_or(a.last().unwrap()))
        }
    };
    get(k) * (1.0 - frac) + get(k + 1.0) * frac
}

/// As [`linear`] with half-cosine transitions (smooth at the centres, so the cubic interpolation
/// between samples is accurate at any fraction).
fn smooth(a: &[f32], sps: f64, offset: f64, x: f64) -> f64 {
    let u = (x - offset) / sps;
    let k = libm::floor(u);
    let w = 0.5 - 0.5 * libm::cos(core::f64::consts::PI * (u - k));
    let at = |t: f64| linear(a, sps, offset, offset + t * sps);
    at(k) * (1.0 - w) + at(k + 1.0) * w
}

fn symbols(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| if rng.bit() { 1.0 } else { -1.0 }).collect()
}

/// A real signal of random `+-1` symbols with linear transitions.
fn real_signal(seed: u64, sps: f64, offset: f64, n_sym: usize) -> Vec<f32> {
    let mut rng = Rng(seed);
    let a = symbols(&mut rng, n_sym);
    let len = (n_sym as f64 * sps) as usize;
    (0..len)
        .map(|n| linear(&a, sps, offset, n as f64) as f32)
        .collect()
}

/// A complex signal of random QPSK symbols `(+-1, +-1)` with linear transitions.
fn qpsk_signal(seed: u64, sps: f64, offset: f64, n_sym: usize) -> Vec<C32> {
    let mut rng = Rng(seed);
    let re = symbols(&mut rng, n_sym);
    let im = symbols(&mut rng, n_sym);
    let len = (n_sym as f64 * sps) as usize;
    (0..len)
        .map(|n| {
            let x = n as f64;
            C32::new(
                linear(&re, sps, offset, x) as f32,
                linear(&im, sps, offset, x) as f32,
            )
        })
        .collect()
}

fn state<T: sdrk_core::Sample>(sps: f64, ted: Ted, ted_gain: f64) -> SymbolSyncState<T> {
    SymbolSyncState::<T>::new(sps, ted, 0.01, DAMP, 0.005, ted_gain).unwrap()
}

/// Runs the kernel `k` over `input` in one call with a generous output.
fn whole<T, K>(k: &K, s: &mut SymbolSyncState<T>, input: &[T]) -> Vec<T>
where
    T: sdrk_core::Sample,
    K: SymbolSync<T>,
{
    let mut out = vec![T::ZERO; input.len() / 2 + 8];
    let p = k.symbol_sync(s, input, &mut out).unwrap();
    assert_eq!(p.consumed, input.len());
    out.truncate(p.produced);
    out
}

/// Runs the kernel over `input` in random pieces with random output sizes (0 to 7), passing the
/// unconsumed input again, as a block does.
fn chunked<T, K>(k: &K, s: &mut SymbolSyncState<T>, input: &[T], seed: u64) -> Vec<T>
where
    T: sdrk_core::Sample,
    K: SymbolSync<T>,
{
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    let mut pos = 0;
    let mut stalls = 0;
    while pos < input.len() {
        let len = rng.below(40).min(input.len() - pos);
        let room = rng.below(8);
        let mut buf = vec![T::ZERO; room];
        let p = k.symbol_sync(s, &input[pos..pos + len], &mut buf).unwrap();
        assert!(p.consumed <= len && p.produced <= room);
        if p.produced < room {
            // Output left over: the whole piece was taken.
            assert_eq!(p.consumed, len);
        }
        out.extend_from_slice(&buf[..p.produced]);
        pos += p.consumed;
        stalls = if p.consumed == 0 { stalls + 1 } else { 0 };
        assert!(stalls < 1000, "no progress");
    }
    // Drain the outputs still possible from the history.
    loop {
        let mut buf = vec![T::ZERO; 4];
        let p = k.symbol_sync(s, &[], &mut buf).unwrap();
        out.extend_from_slice(&buf[..p.produced]);
        if p.produced == 0 {
            break;
        }
    }
    out
}

fn bits_c(v: &[C32]) -> Vec<(u32, u32)> {
    v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

fn bits_r(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn construction_rules() {
    let ok = |sps, bw, damp, dev, gain| {
        SymbolSyncState::<C32>::new(sps, Ted::Gardner, bw, damp, dev, gain).is_ok()
    };
    assert!(ok(2.0, 0.0001, 0.1, 0.0, 1e-9));
    assert!(ok(1_000_000.0, 0.1, 4.0, 0.05, 1e9));
    assert!(ok(4.1, 0.01, DAMP, 0.005, 1.0));
    for (sps, bw, damp, dev, gain) in [
        (1.999, 0.01, DAMP, 0.005, 1.0),
        (1_000_000.5, 0.01, DAMP, 0.005, 1.0),
        (f64::NAN, 0.01, DAMP, 0.005, 1.0),
        (f64::INFINITY, 0.01, DAMP, 0.005, 1.0),
        (4.0, 0.000_099, DAMP, 0.005, 1.0),
        (4.0, 0.100_01, DAMP, 0.005, 1.0),
        (4.0, f64::NAN, DAMP, 0.005, 1.0),
        (4.0, 0.01, 0.099, 0.005, 1.0),
        (4.0, 0.01, 4.01, 0.005, 1.0),
        (4.0, 0.01, f64::NAN, 0.005, 1.0),
        (4.0, 0.01, DAMP, -0.001, 1.0),
        (4.0, 0.01, DAMP, 0.0501, 1.0),
        (4.0, 0.01, DAMP, f64::NAN, 1.0),
        (4.0, 0.01, DAMP, 0.005, 0.0),
        (4.0, 0.01, DAMP, 0.005, -1.0),
        (4.0, 0.01, DAMP, 0.005, f64::NAN),
        (4.0, 0.01, DAMP, 0.005, f64::INFINITY),
        // The gains overflow.
        (4.0, 0.01, DAMP, 0.005, 1e-310),
    ] {
        let e = SymbolSyncState::<C32>::new(sps, Ted::Gardner, bw, damp, dev, gain).unwrap_err();
        assert!(
            matches!(e, Error::InvalidParameter(_)),
            "{sps} {bw} {damp} {dev} {gain}"
        );
        assert_eq!(e.kind(), ErrorKind::InvalidArgument);
    }
    for e in [
        SymbolSyncState::<CQ15>::new(4.0, Ted::Gardner, 0.01, DAMP, 0.005, 1.0).unwrap_err(),
        SymbolSyncState::<Q15>::new(4.0, Ted::MuellerMuller, 0.01, DAMP, 0.005, 1.0).unwrap_err(),
    ] {
        assert!(matches!(e, Error::Unsupported(_)));
    }
}

#[test]
fn loop_constants_and_getters() {
    let s = SymbolSyncState::<f32>::new(8.0, Ted::MuellerMuller, 0.02, 0.5, 0.01, 2.0).unwrap();
    let theta = 0.02 / (0.5 + 1.0 / (4.0 * 0.5));
    let d = (1.0 + (2.0 * 0.5) * theta) + theta * theta;
    assert_eq!(
        s.gains(),
        SymbolSyncGains {
            kp: ((((4.0 * 0.5) * theta) / d) / 2.0) * 8.0,
            ki: (((4.0 * (theta * theta)) / d) / 2.0) * 8.0,
            max_integ: 0.01 * 8.0,
        }
    );
    assert_eq!(
        (
            s.sps(),
            s.ted(),
            s.loop_bw(),
            s.damping(),
            s.max_deviation(),
            s.ted_gain()
        ),
        (8.0, Ted::MuellerMuller, 0.02, 0.5, 0.01, 2.0)
    );
    assert_eq!(s.sps_estimate(), 8.0);
}

#[test]
fn timing() {
    let s = state::<C32>(4.1, Ted::Gardner, 1.0);
    assert_eq!(s.ratio(), Ratio::new(10, 41).unwrap());
    assert_eq!(s.ratio_kind(), RatioKind::Variable);
    assert_eq!(s.group_delay(), Delay::new(1, 1).unwrap());
    // 48 kHz / 1187.5 Bd: the kernel rounds 1 / sps to a denominator of 10^6 / gcd.
    let s = state::<f32>(48_000.0 / 1187.5, Ted::Gardner, 1.0);
    assert_eq!(s.ratio(), Ratio::new(1_000_000, 40_421_053).unwrap());
    assert_eq!(
        state::<f32>(2.0, Ted::Gardner, 1.0).ratio(),
        Ratio::new(1, 2).unwrap()
    );
}

/// With a detector gain of `1e290` the loop gains are about `1e-292`: the steps vanish in the
/// rounding, so the strobes stay at `1 + k sps` exactly whatever the input.
#[test]
fn interpolation_at_mu_zero_returns_the_input_sample() {
    let mut rng = Rng(381);
    let x: Vec<f32> = (0..4001).map(|_| rng.next() * 3.0).collect();
    let xc: Vec<C32> = (0..4001)
        .map(|_| C32::new(rng.next(), -rng.next()))
        .collect();
    for ted in [Ted::Gardner, Ted::MuellerMuller] {
        for sps in [2.0, 3.0, 4.0, 7.0] {
            let mut s = state::<f32>(sps, ted, 1e290);
            let y = whole(&Fallback, &mut s, &x);
            let step = sps as usize;
            let expect: Vec<f32> = x[1..x.len() - 2].iter().step_by(step).copied().collect();
            assert_eq!(bits_r(&y), bits_r(&expect), "{ted:?} {sps}");
            assert_eq!(s.sps_estimate(), sps);
            let mut s = state::<C32>(sps, ted, 1e290);
            let y = whole(&Fallback, &mut s, &xc);
            let expect: Vec<C32> = xc[1..xc.len() - 2].iter().step_by(step).copied().collect();
            assert_eq!(bits_c(&y), bits_c(&expect), "{ted:?} {sps}");
        }
    }
}

/// At `sps = 2.5` with the loop frozen the strobes are at `1, 3.5, 6, 8.5, ...`: `mu` alternates
/// between 0 and 0.5, and the cubic interpolation reproduces a cubic polynomial (to f32
/// rounding).
#[test]
fn interpolation_is_exact_on_a_cubic() {
    let p = |x: f64| 0.3 + 0.02 * x - 0.0004 * x * x + 0.000_002 * x * x * x;
    let x: Vec<f32> = (0..200).map(|n| p(f64::from(n)) as f32).collect();
    let mut s = state::<f32>(2.5, Ted::Gardner, 1e290);
    let y = whole(&Fallback, &mut s, &x);
    assert_eq!(y.len(), 79);
    for (k, &v) in y.iter().enumerate() {
        let t = 1.0 + 2.5 * k as f64;
        assert!((f64::from(v) - p(t)).abs() < 2e-6, "{k} {v} {}", p(t));
    }
}

#[test]
fn constant_input_gives_the_constant() {
    for c in [0.0f32, 0.7, -1.0] {
        let x = vec![c; 1000];
        for ted in [Ted::Gardner, Ted::MuellerMuller] {
            let mut s = state::<f32>(4.0, ted, 1.0);
            let y = whole(&Fallback, &mut s, &x);
            // Strobes at 1, 5, ..., 997 (the last needs x[999]).
            assert_eq!(y.len(), 250);
            assert!(y.iter().all(|&v| v == c), "{c} {ted:?}");
            assert_eq!(s.sps_estimate(), 4.0);
            let xc = vec![C32::new(c, -c); 1000];
            let mut s = state::<C32>(4.0, ted, 1.0);
            let y = whole(&Fallback, &mut s, &xc);
            assert_eq!(y.len(), 250);
            assert!(y.iter().all(|&v| v == C32::new(c, -c)));
        }
    }
}

/// The largest distance of the last `n` outputs from a symbol value `+-1` per component.
fn worst_real(y: &[f32], n: usize) -> f32 {
    y[y.len() - n..]
        .iter()
        .map(|v| (v.abs() - 1.0).abs())
        .fold(0.0, f32::max)
}

fn worst_complex(y: &[C32], n: usize) -> f32 {
    y[y.len() - n..]
        .iter()
        .map(|v| (v.re.abs() - 1.0).abs().max((v.im.abs() - 1.0).abs()))
        .fold(0.0, f32::max)
}

/// Clean signals with linear transitions, the first symbol centre half a symbol off the first
/// strobe: after 1000 symbols every strobe lies within 0.01 symbols of a centre (the value at
/// `tau` symbols from a centre between two different symbols is `1 - 2 tau`). The detector
/// gains are the slopes of the detectors for these signals per symbol (2 for Gardner with half
/// of the symbols changing, 1 for Mueller–Müller), so the loop has its nominal bandwidth at
/// every `sps` (D565).
#[test]
fn clean_signals_at_integer_sps_lock() {
    for sps in [2.0, 4.0, 8.0] {
        let offset = 1.0 + 0.5 * sps;
        let x = real_signal(11, sps, offset, 1300);
        let mut s = state::<f32>(sps, Ted::Gardner, 2.0);
        let y = whole(&Fallback, &mut s, &x);
        assert!(
            worst_real(&y, 200) < 0.02,
            "gardner f32 {sps}: {}",
            worst_real(&y, 200)
        );
        assert!((s.sps_estimate() - sps).abs() < 1e-3 * sps);

        let x = qpsk_signal(12, sps, offset, 1300);
        let mut s = state::<C32>(sps, Ted::Gardner, 4.0);
        let y = whole(&Fallback, &mut s, &x);
        assert!(
            worst_complex(&y, 200) < 0.02,
            "gardner c32 {sps}: {}",
            worst_complex(&y, 200)
        );

        // Mueller–Müller has a hang point half a symbol off (a strobe on a transition gives
        // `y = 0` and `d = 0`), so its signals start a quarter of a symbol off; at sps 2 that is
        // no whole sample (the linear transitions interpolate exactly only at whole samples), so
        // there the signal starts on time and must stay there.
        let offset = if sps == 2.0 { 1.0 } else { 1.0 + 0.25 * sps };
        let x = qpsk_signal(12, sps, offset, 1300);
        let mut s = state::<C32>(sps, Ted::MuellerMuller, 2.0);
        let y = whole(&Fallback, &mut s, &x);
        assert!(
            worst_complex(&y, 200) < 0.02,
            "mm c32 {sps}: {}",
            worst_complex(&y, 200)
        );

        let x = real_signal(13, sps, offset, 1300);
        let mut s = state::<f32>(sps, Ted::MuellerMuller, 1.0);
        let y = whole(&Fallback, &mut s, &x);
        assert!(
            worst_real(&y, 200) < 0.02,
            "mm f32 {sps}: {}",
            worst_real(&y, 200)
        );
        // Half a symbol off is the hang point: a strobe on every transition.
        let x = real_signal(13, sps, 1.0 + 0.5 * sps, 1300);
        let mut s = state::<f32>(sps, Ted::MuellerMuller, 1.0);
        let y = whole(&Fallback, &mut s, &x);
        assert!(worst_real(&y, 200) > 0.5, "mm f32 {sps} hang point");
    }
}

/// A clock 0.3 % slow at sps 4 (4.012 samples per symbol): the integrator takes up the
/// difference. Half-cosine transitions, so that the strobes at every fraction interpolate well;
/// a strobe `tau` symbols from a centre between two different symbols gives `cos(pi tau)`, so
/// 0.03 bounds `tau` by 0.08 symbols (the interpolation error of the half-cosine at 4 samples
/// per symbol is about 0.015).
#[test]
fn the_integrator_tracks_a_rate_offset() {
    let sps = 4.0 * 1.003;
    let mut rng = Rng(21);
    let a = symbols(&mut rng, 3000);
    let x: Vec<f32> = (0..12_000)
        .map(|n| smooth(&a, sps, 2.0, f64::from(n)) as f32)
        .collect();
    let mut s = state::<f32>(4.0, Ted::Gardner, 2.0);
    let y = whole(&Fallback, &mut s, &x);
    assert!(worst_real(&y, 500) < 0.03, "{}", worst_real(&y, 500));
    assert!(
        (s.sps_estimate() - sps).abs() < 1e-3,
        "{}",
        s.sps_estimate()
    );
    // The integrator stops at max_deviation * sps.
    let x = real_signal(22, 4.0 * 1.02, 3.0, 2000);
    let mut s = state::<f32>(4.0, Ted::Gardner, 2.0);
    let mut out = [0.0f32; 1];
    let mut highest = 0.0f64;
    let mut pos = 0;
    while pos < x.len() {
        pos += Fallback
            .symbol_sync(&mut s, &x[pos..], &mut out)
            .unwrap()
            .consumed;
        assert!(s.sps_estimate() <= 4.0 + 0.005 * 4.0);
        highest = highest.max(s.sps_estimate());
    }
    assert_eq!(highest, 4.0 + 0.005 * 4.0);
}

fn noisy_real(seed: u64, sps: f64) -> Vec<f32> {
    let mut x = real_signal(seed, sps, 2.3, 1500);
    let mut rng = Rng(seed + 100);
    for v in &mut x {
        *v += 0.2 * rng.next();
    }
    x
}

fn noisy_qpsk(seed: u64, sps: f64) -> Vec<C32> {
    let mut x = qpsk_signal(seed, sps, 2.3, 1500);
    let mut rng = Rng(seed + 100);
    for v in &mut x {
        *v += C32::new(0.2 * rng.next(), 0.2 * rng.next());
    }
    x
}

#[test]
fn chunking_does_not_change_the_output() {
    for ted in [Ted::Gardner, Ted::MuellerMuller] {
        for (i, sps) in [2.0, 4.1, 7.3].into_iter().enumerate() {
            let x = noisy_real(30 + i as u64, sps);
            let mut s = state::<f32>(sps, ted, 2.0);
            let a = whole(&Fallback, &mut s, &x);
            let end = s;
            for plan in 0..5 {
                let mut s = state::<f32>(sps, ted, 2.0);
                let b = chunked(&Fallback, &mut s, &x, 700 + plan);
                assert_eq!(bits_r(&a), bits_r(&b), "{ted:?} {sps} {plan}");
                assert_eq!(s, end);
            }
            let x = noisy_qpsk(40 + i as u64, sps);
            let mut s = state::<C32>(sps, ted, 2.0);
            let a = whole(&Fallback, &mut s, &x);
            let end = s;
            for plan in 0..5 {
                let mut s = state::<C32>(sps, ted, 2.0);
                let b = chunked(&Fallback, &mut s, &x, 800 + plan);
                assert_eq!(bits_c(&a), bits_c(&b), "{ted:?} {sps} {plan}");
                assert_eq!(s, end);
            }
        }
    }
}

#[test]
fn a_full_output_leaves_the_rest_of_the_input() {
    let x = noisy_real(50, 4.0);
    let mut s = state::<f32>(4.0, Ted::Gardner, 2.0);
    let mut out = [0.0f32; 10];
    let p = Fallback.symbol_sync(&mut s, &x, &mut out).unwrap();
    assert_eq!(p.produced, 10);
    // Strobe 10 is near input sample 41 and its mid sample near 43: the window of the mid
    // sample starts near 42, and everything before it is consumed.
    assert!((38..=44).contains(&p.consumed), "{p:?}");
    let mut empty: [f32; 0] = [];
    let q = Fallback
        .symbol_sync(&mut s, &x[p.consumed..], &mut empty)
        .unwrap();
    assert_eq!(
        q,
        Progress {
            consumed: 0,
            produced: 0
        }
    );
    // Short input: everything is taken into the history, nothing comes out yet.
    let mut s = state::<f32>(4.0, Ted::Gardner, 2.0);
    let p = Fallback.symbol_sync(&mut s, &x[..3], &mut out).unwrap();
    assert_eq!(
        p,
        Progress {
            consumed: 3,
            produced: 0
        }
    );
    let p = Fallback.symbol_sync(&mut s, &x[3..4], &mut out).unwrap();
    assert_eq!(
        p,
        Progress {
            consumed: 1,
            produced: 1
        }
    );
    assert_eq!(out[0], x[1]);
}

#[test]
fn reset_restarts_the_loop() {
    let x = noisy_qpsk(60, 4.1);
    for ted in [Ted::Gardner, Ted::MuellerMuller] {
        let fresh = state::<C32>(4.1, ted, 2.0);
        let mut s = fresh;
        let a = whole(&Fallback, &mut s, &x);
        assert_ne!(s, fresh);
        s.reset();
        assert_eq!(s, fresh);
        let b = whole(&Fallback, &mut s, &x);
        assert_eq!(bits_c(&a), bits_c(&b));
    }
}

/// NaN and infinity in the input: the error of the strobes they reach counts as zero, so the
/// loop keeps running and the outputs are finite again after the bad samples.
#[test]
fn non_finite_input_does_not_derail_the_loop() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        let mut x = real_signal(70, 4.0, 3.0, 1500);
        for v in &mut x[2000..2040] {
            *v = bad;
        }
        let mut s = state::<f32>(4.0, Ted::Gardner, 2.0);
        let y = whole(&Fallback, &mut s, &x);
        assert!((1495..=1505).contains(&y.len()), "{bad} {}", y.len());
        assert!(s.sps_estimate().is_finite());
        assert!(worst_real(&y, 200) < 0.02, "{bad}");
        let mut xc = qpsk_signal(71, 4.0, 2.0, 1500);
        for v in &mut xc[3000..3010] {
            *v = C32::new(bad, -bad);
        }
        let mut s = state::<C32>(4.0, Ted::MuellerMuller, 2.0);
        let y = whole(&Fallback, &mut s, &xc);
        assert!((1495..=1505).contains(&y.len()));
        assert!(worst_complex(&y, 200) < 0.02, "{bad}");
    }
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send<T: Send + Sync>() {}
    assert_send::<Ted>();
    assert_send::<SymbolSyncGains>();
    assert_send::<SymbolSyncState<C32>>();
    assert_send::<SymbolSyncState<f32>>();
}

#[cfg(feature = "reference")]
mod against_the_reference {
    use super::*;
    use sdrk_core::Reference;

    #[test]
    fn fallback_equals_reference() {
        for ted in [Ted::Gardner, Ted::MuellerMuller] {
            for (i, sps) in [2.0, 4.1, 10.1].into_iter().enumerate() {
                let mut x = noisy_real(80 + i as u64, sps);
                x[500] = f32::NAN;
                x[900] = f32::INFINITY;
                let mut a = state::<f32>(sps, ted, 2.0);
                let mut b = a;
                let ya = whole(&Fallback, &mut a, &x);
                let yb = whole(&Reference, &mut b, &x);
                assert_eq!(bits_r(&ya), bits_r(&yb), "{ted:?} {sps}");
                assert_eq!(a, b);
                let mut b = state::<f32>(sps, ted, 2.0);
                let yc = chunked(&Reference, &mut b, &x, 900 + i as u64);
                assert_eq!(bits_r(&ya), bits_r(&yc), "{ted:?} {sps} chunked");

                let mut x = noisy_qpsk(90 + i as u64, sps);
                x[700] = C32::new(f32::NAN, 0.0);
                let mut a = state::<C32>(sps, ted, 2.0);
                let mut b = a;
                let ya = whole(&Fallback, &mut a, &x);
                let yb = whole(&Reference, &mut b, &x);
                assert_eq!(bits_c(&ya), bits_c(&yb), "{ted:?} {sps}");
                assert_eq!(a, b);
                let mut b = state::<C32>(sps, ted, 2.0);
                let yc = chunked(&Reference, &mut b, &x, 950 + i as u64);
                assert_eq!(bits_c(&ya), bits_c(&yc), "{ted:?} {sps} chunked");
            }
        }
    }
}
