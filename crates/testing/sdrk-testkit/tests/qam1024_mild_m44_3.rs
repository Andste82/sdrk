//! M44.3 §3, information: 1024-QAM through `Multipath::mild` (I44.3-2, moved to M44.4 by
//! D2422). `qam_mod` (4 samples per symbol) -> `Multipath::mild(4.0)` -> Gaussian noise at
//! Eb/N0 27 and 29 dB (the points of test 3 of M44.2) -> `qam_demod` with `equalizer_taps` 15,
//! 80000 symbols, seed 894.
//! With the blind start of D1517 (16000 CMA symbols with `mu_cma` 0.005) these receivers stay in
//! a false state without lock (the steps of M44.3 found none of 16 receivers at lock, D2411); the
//! test prints what each receiver does (the symbol of its first lock, its restarts, the decision
//! error of its equalizer relative to the squared half level spacing and its symbol error rate
//! over the last 10000 symbols) and guards only what must hold whatever the outcome: a receiver
//! that reports lock at the end decides its last 10000 symbols correctly (symbol error rate below
//! 1 %), so a false state is never reported as lock. Since the multi-modulus start of M44.4 §3
//! the receivers lock, and the one at Eb/N0 27 dB reads 1.65 % at the floor of the channel, above
//! the bound: open item I44.4-2 (D2513).
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::print_stderr,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
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

/// The seed of the test (signal and noise).
const SEED: u64 = 894;
/// Samples per symbol.
const SPS: usize = 4;
/// Symbol rate in Hz.
const SYMBOL_RATE: u64 = 10_000;
/// Bits per 1024-QAM symbol.
const BITS: usize = 10;
/// Symbols sent.
const SYMBOLS: usize = 80_000;
/// Symbols at the end over which the symbol error rate is counted.
const TAIL: usize = 10_000;
/// Samples per call of the receivers.
const CALL: usize = 4096;

/// The symbols (10 bits each, the first the most significant) of the soft bits `soft`.
fn symbols(soft: &[f32]) -> Vec<u32> {
    soft.chunks_exact(BITS)
        .map(|b| b.iter().fold(0u32, |g, &s| (g << 1) | u32::from(s > 0.0)))
        .collect()
}

/// The symbol error rate of the last [`TAIL`] received symbols at the best rotation and lag.
fn tail_ser(sent: &[u32], got: &[u32]) -> f64 {
    let from = got.len().saturating_sub(TAIL);
    let mut best = 1.0f64;
    for r in 0..4u32 {
        for lag in 0..128usize {
            let wrong = (from..got.len())
                .filter(|&j| j >= lag && rotate_symbol(got[j], 5, (4 - r) % 4) != sent[j - lag])
                .count();
            best = best.min(wrong as f64 / (got.len() - from).max(1) as f64);
        }
    }
    best
}

#[test]
fn qam1024_through_mild_multipath_is_never_reported_locked_in_a_false_state() {
    let mut rng = Pcg32::new(SEED, 1);
    let bits: Vec<f32> = (0..SYMBOLS * BITS)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect();
    let sent = symbols(&bits);
    let tx_cfg = QamModConfig {
        constellation: SymbolMapConstellation::Qam1024,
        samples_per_symbol: SPS,
        ..QamModConfig::default()
    };
    let rate = Rate::hz(SYMBOL_RATE * BITS as u64).unwrap();
    let mut tx = QamModBlock::<VecStorage>::new(&tx_cfg, rate, &mut VecStorage).unwrap();
    let mut sig = vec![C32::default(); SYMBOLS * SPS];
    assert_eq!(tx.process(&bits, &mut sig).unwrap().produced, sig.len());
    let wide: Vec<Complex64> = sig
        .iter()
        .map(|c| Complex64::new(f64::from(c.re), f64::from(c.im)))
        .collect();
    let mut faded = Vec::new();
    Multipath::mild(SPS as f64)
        .unwrap()
        .process(&wide, &mut faded);
    let power =
        faded[1000..].iter().map(Complex64::norm_sqr).sum::<f64>() / (faded.len() - 1000) as f64;
    // The squared half level spacing `1 / E` of 1024-QAM at unit power.
    let h2 = (1.0 / Qam::Qam1024.energy()) as f32;
    for ebn0 in [27.0, 29.0] {
        let mut noisy = Vec::new();
        Awgn::from_ebn0(ebn0, BITS as u32, SPS as f64, power, SEED)
            .unwrap()
            .process(&faded, &mut noisy);
        let input: Vec<C32> = noisy
            .iter()
            .map(|c| C32::new(c.re as f32, c.im as f32))
            .collect();
        let cfg = QamDemodConfig {
            constellation: SymbolMapConstellation::Qam1024,
            symbol_rate_hz: SYMBOL_RATE as f64,
            equalizer_taps: 15,
            ..QamDemodConfig::default()
        };
        let mut rx = QamDemodBlock::new(
            &cfg,
            Rate::hz(SYMBOL_RATE * SPS as u64).unwrap(),
            &mut VecStorage,
        )
        .unwrap();
        let mut soft = Vec::new();
        let mut out = vec![0.0f32; CALL * BITS];
        let mut first_lock = None;
        for piece in input.chunks(CALL) {
            let mut at = 0;
            while at < piece.len() {
                let p = rx.process(&piece[at..], &mut out).unwrap();
                soft.extend_from_slice(&out[..p.produced]);
                at += p.consumed;
            }
            if first_lock.is_none() && rx.locked() {
                first_lock = Some(soft.len() / BITS);
            }
        }
        let got = symbols(&soft);
        let ser = tail_ser(&sent, &got);
        let mse = rx.equalizer().map_or(f32::NAN, |e| e.mse() / h2);
        eprintln!(
            "1024-QAM mild Eb/N0 {ebn0} dB, 15 taps: first lock at symbol {first_lock:?}, locked at \
             the end {}, restarts {}, decision error / h^2 {mse:.3}, symbol error rate of the last \
             {TAIL} symbols {ser:.4}",
            rx.locked(),
            rx.restarts()
        );
        assert!(
            !rx.locked() || ser < 0.01,
            "Eb/N0 {ebn0} dB: reported lock with a symbol error rate of {ser}"
        );
    }
}
