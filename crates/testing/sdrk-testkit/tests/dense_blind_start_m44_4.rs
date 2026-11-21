//! M44.4 test 3, the blind start of the dense grids through multipath (§3): `qam_mod` (4 samples
//! per symbol) -> `Multipath::mild(4.0)` -> Gaussian noise -> `qam_demod` with
//! `equalizer_taps` 15 (the flow `qam256_receiver`), in calls of 512 symbols; the equalizer of
//! 256-QAM and 1024-QAM starts with the multi-modulus cost (D2507 to D2509).
//!
//! - 1024-QAM at Eb/N0 29 dB (seed 904, 16 receivers, 80000 symbols): at least 15 of 16
//!   receivers lock by symbol 70000 and stay locked, with a symbol error rate of the last 10000
//!   symbols below 1 %; a receiver that locks later stays below 5 % after its lock (a false state
//!   is never reported as lock). With the constant-modulus start of M44.3 none of these receivers
//!   locked (D2510).
//! - 256-QAM at Eb/N0 22 and 24 dB (seed 905, 8 receivers each, 60000 symbols): every receiver
//!   locks, no later than with the constant-modulus start of M44.3 ([`M44_3_LOCK`], measured on
//!   the same signals with the code of M44.3, D2510), stays locked to the end and decides its
//!   last 10000 symbols correctly (below 1 %).
//!
//! The lock symbol of a receiver is the symbol count at the end of the first call after which it
//! stays locked to the end of the signal.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::print_stderr,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use num_complex::Complex64;
use sdrk::Rate;
use sdrk_blocks::{
    Block, QamDemodBlock, QamDemodConfig, QamModBlock, QamModConfig, SymbolMapConstellation,
};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::{Qam, rotate_symbol};
use sdrk_testkit::channel::{Awgn, Impairment, Multipath};
use sdrk_testkit::{Pcg32, VecStorage};

/// The seed of the 1024-QAM receivers.
const SEED_1024: u64 = 904;
/// The seed of the 256-QAM receivers.
const SEED_256: u64 = 905;
/// Samples per symbol.
const SPS: usize = 4;
/// Symbol rate in Hz.
const SYMBOL_RATE: u64 = 10_000;
/// Symbols per call of the receivers.
const CALL_SYMBOLS: usize = 512;
/// Symbols at the end over which the symbol error rate is counted.
const TAIL: usize = 10_000;

/// The lock symbols of the 256-QAM receivers with the blind start of M44.3 (constant modulus,
/// `mu_cma` 0.02 over 8000 symbols), measured on the signals of this test with the code of
/// M44.3 (D2510): Eb/N0 22 dB receivers 0 to 7, then 24 dB receivers 0 to 7; `None`: no lock
/// within the 60000 symbols.
const M44_3_LOCK: [[Option<u64>; 8]; 2] = [
    [
        Some(18944),
        Some(23040),
        Some(13824),
        Some(17408),
        None,
        Some(15872),
        Some(17408),
        Some(18432),
    ],
    [
        Some(15360),
        Some(31232),
        Some(12800),
        Some(15360),
        Some(32768),
        Some(14848),
        Some(16896),
        Some(15872),
    ],
];

fn constellation_of(q: Qam) -> SymbolMapConstellation {
    match q {
        Qam::Qam256 => SymbolMapConstellation::Qam256,
        _ => SymbolMapConstellation::Qam1024,
    }
}

/// The symbols (`bits` bits each, the first the most significant) of the soft bits `soft`.
fn symbols(soft: &[f32], bits: usize) -> Vec<u32> {
    soft.chunks_exact(bits)
        .map(|b| b.iter().fold(0u32, |g, &s| (g << 1) | u32::from(s > 0.0)))
        .collect()
}

/// The signal of receiver `rx`: random bits of the stream of `rx` through `qam_mod`,
/// `Multipath::mild` and Gaussian noise at `ebn0` (the power of the faded signal after its first
/// quarter); returns the input and the sent symbols.
fn signal(q: Qam, seed: u64, rx: u64, ebn0: f64, n_sym: usize) -> (Vec<C32>, Vec<u32>) {
    let bits = q.bits_per_symbol();
    let stream = 100 * q.bits_per_axis() as u64 + 10 * rx + 1;
    let mut rng = Pcg32::new(seed, 7000 + stream);
    let soft_tx: Vec<f32> = (0..(n_sym + 64) * bits)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect();
    let sent = symbols(&soft_tx, bits);
    let cfg = QamModConfig {
        constellation: constellation_of(q),
        samples_per_symbol: SPS,
        ..QamModConfig::default()
    };
    let rate = Rate::hz(SYMBOL_RATE * bits as u64).unwrap();
    let mut tx = QamModBlock::<VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap();
    let mut sig = vec![C32::default(); n_sym * SPS];
    let p = tx.process(&soft_tx, &mut sig).unwrap();
    sig.truncate(p.produced);
    let wide: Vec<Complex64> = sig
        .iter()
        .map(|c| Complex64::new(f64::from(c.re), f64::from(c.im)))
        .collect();
    let mut faded = Vec::new();
    Multipath::mild(SPS as f64)
        .unwrap()
        .process(&wide, &mut faded);
    let body = &faded[faded.len() / 4..];
    let power = body.iter().map(Complex64::norm_sqr).sum::<f64>() / body.len() as f64;
    let mut noisy = Vec::new();
    Awgn::from_ebn0(ebn0, bits as u32, SPS as f64, power, seed * 1000 + stream)
        .unwrap()
        .process(&faded, &mut noisy);
    let input = noisy
        .iter()
        .map(|c| C32::new(c.re as f32, c.im as f32))
        .collect();
    (input, sent)
}

/// What a receiver did: the lock symbol, whether it is locked at the end, and the symbol error
/// rate of the last [`TAIL`] symbols (of those after the lock when it came later) at the best
/// quarter turn and lag.
struct Outcome {
    lock: Option<u64>,
    locked: bool,
    ser: f64,
}

fn receive(q: Qam, input: &[C32], sent: &[u32]) -> Outcome {
    let bits = q.bits_per_symbol();
    let cfg = QamDemodConfig {
        constellation: constellation_of(q),
        symbol_rate_hz: SYMBOL_RATE as f64,
        equalizer_taps: 15,
        ..QamDemodConfig::default()
    };
    let rate = Rate::hz(SYMBOL_RATE * SPS as u64).unwrap();
    let mut rx = QamDemodBlock::new(&cfg, rate, &mut VecStorage).unwrap();
    let mut out = vec![0.0f32; 1 << 16];
    let mut soft = Vec::new();
    let mut lock = None;
    for piece in input.chunks(CALL_SYMBOLS * SPS) {
        let mut rest = piece;
        while !rest.is_empty() {
            let p = rx.process(rest, &mut out).unwrap();
            soft.extend_from_slice(&out[..p.produced]);
            rest = &rest[p.consumed..];
        }
        if rx.locked() {
            lock.get_or_insert(rx.symbols());
        } else {
            lock = None;
        }
    }
    let got = symbols(&soft, bits);
    let half = q.bits_per_axis() as u32;
    // The last `TAIL` symbols, or those after the lock when it came later (the symbols of the
    // receiver lag those of `rx.symbols()` by the delay of the equalizer, a few symbols).
    let locked_from = lock.map_or(0, |l| usize::try_from(l).unwrap());
    let from = (got.len() - TAIL).max(locked_from.min(got.len() - 1000));
    let mut best = usize::MAX;
    for r in 0..4u32 {
        for lag in 0..128usize {
            let wrong = (from..got.len())
                .filter(|&j| j < lag || rotate_symbol(got[j], half, (4 - r) % 4) != sent[j - lag])
                .count();
            best = best.min(wrong);
        }
    }
    Outcome {
        lock,
        locked: rx.locked(),
        ser: best as f64 / (got.len() - from) as f64,
    }
}

/// The receivers `(ebn0, rx)` of `q`, each on its own thread.
fn run_all(q: Qam, seed: u64, jobs: &[(f64, u64)], n_sym: usize) -> Vec<Outcome> {
    std::thread::scope(|s| {
        let handles: Vec<_> = jobs
            .iter()
            .map(|&(ebn0, rx)| {
                s.spawn(move || {
                    let (input, sent) = signal(q, seed, rx, ebn0, n_sym);
                    receive(q, &input, &sent)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

#[test]
fn qam1024_through_mild_multipath_locks_in_15_of_16_receivers() {
    let n_sym = 80_000;
    let jobs: Vec<(f64, u64)> = (0..16).map(|rx| (29.0, rx)).collect();
    let outcomes = run_all(Qam::Qam1024, SEED_1024, &jobs, n_sym);
    let mut held = 0;
    for (&(ebn0, rx), o) in jobs.iter().zip(&outcomes) {
        eprintln!(
            "1024-QAM mild Eb/N0 {ebn0} dB receiver {rx}: lock {:?}, locked at the end {}, \
             symbol error rate after the lock (at most the last {TAIL} symbols) {:.4}",
            o.lock, o.locked, o.ser
        );
        if let Some(lock) = o.lock {
            // Never a false state reported as lock (those decide 60 % of the symbols wrong or
            // more): a receiver that locks late, still settling, stays below 5 %.
            assert!(
                o.ser < 0.05,
                "receiver {rx}: lock at {lock} with a symbol error rate of {}",
                o.ser
            );
            // Locked over the whole tail: the floor of the channel, below 1 %.
            if lock as usize + TAIL <= n_sym {
                assert!(
                    o.ser < 0.01,
                    "receiver {rx}: symbol error rate {} in lock",
                    o.ser
                );
                held += 1;
            }
        }
    }
    assert!(
        held >= 15,
        "{held} of 16 receivers locked by symbol {}",
        n_sym - TAIL
    );
}

#[test]
fn qam256_through_mild_multipath_locks_no_later_than_with_the_constant_modulus_start() {
    let jobs: Vec<(f64, u64)> = [22.0, 24.0]
        .into_iter()
        .flat_map(|e| (0..8).map(move |rx| (e, rx)))
        .collect();
    let outcomes = run_all(Qam::Qam256, SEED_256, &jobs, 60_000);
    for (k, (&(ebn0, rx), o)) in jobs.iter().zip(&outcomes).enumerate() {
        let before = M44_3_LOCK[k / 8][k % 8];
        eprintln!(
            "256-QAM mild Eb/N0 {ebn0} dB receiver {rx}: lock {:?} (M44.3 {before:?}), symbol \
             error rate after the lock (at most the last {TAIL} symbols) {:.4}",
            o.lock, o.ser
        );
        let lock = o
            .lock
            .unwrap_or_else(|| panic!("Eb/N0 {ebn0} dB receiver {rx}: no lock"));
        assert!(o.locked);
        assert!(
            before.is_none_or(|b| lock <= b),
            "Eb/N0 {ebn0} dB receiver {rx}: lock at {lock}, with M44.3 at {before:?}"
        );
        assert!(
            o.ser < 0.01,
            "Eb/N0 {ebn0} dB receiver {rx}: symbol error rate {}",
            o.ser
        );
    }
}
