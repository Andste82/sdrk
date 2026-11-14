//! The restart of the frequency acquisition of `qam_demod` below the lock level (I44-9, the
//! maintainer's decision D1431, implemented by D1432): `qam_mod` -> carrier offset 0.001 cycles
//! per symbol -> noise -> `qam_demod` with its defaults (`equalizer_taps` 11, `carrier_bw` 0.005)
//! and without the equalizer.
//!
//! - **A correct estimate is kept** (seed 780): 16-QAM at Es/N0 11.5 and 12 dB and 64-QAM at
//!   18 dB, below the level at which the lock detector of `costas` turns on (16-QAM about
//!   14.8 dB, 64-QAM about 21.0 dB), over 100000 symbols (24 checks): the receiver never
//!   restarts, its estimate is within 1e-4 of the offset, the received symbols keep one
//!   alignment with the sent ones (every window of 2000 symbols has its fewest bit errors at the
//!   rotation and lag of the first 400 counted symbols, among the four rotations and that lag
//!   +-3 symbols: no lost symbol, no new rotation), and the BER is at most the Gray
//!   approximation of `qam_ber_m44` at Es/N0 1.5 dB lower. Before D1432 the check of D1421 restarted these receivers every 4864
//!   symbols (61 restarts in 300000 symbols, BER 0.12 to 0.24 measured per window of 2000
//!   symbols at the best alignment).
//! - **A wrong estimate is dropped** (seed 781): a carrier at 0.06 cycles per symbol (its
//!   fourth power is a clear peak, so the first estimate is the carrier's), then either the weak
//!   signal directly (the loops on a wrong frequency) or first Gaussian noise of the signal's
//!   power for 20480 symbols (the noise-first start, the loops on noise), then the weak signal:
//!   the receiver restarts at least once, ends with the estimate within 1e-4 of the offset, and
//!   the last 20000 symbols keep one alignment within the BER bound above.
//! - **An estimate at the edge of the range is kept** (seeds 782 and 783, D1435): 16-QAM at
//!   Es/N0 12 dB with the equalizer at +-0.12497 cycles per symbol, where the estimate of a
//!   verify window can wrap to the other edge of `[-0.125, 0.125)`: no restart over 40000
//!   symbols, the estimate within 1e-4 modulo a quarter cycle per symbol.
//!
//! The measured values are printed (run with `--nocapture`).
#![cfg(feature = "std")]
#![allow(
    clippy::cast_lossless,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::print_stderr,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

use num_complex::Complex64;
use sdrk::Rate;
use sdrk_blocks::{Block, QamDemodBlock, QamDemodConfig, QamModBlock, QamModConfig};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::rotate_symbol;
use sdrk_testkit::channel::{Awgn, FrequencyOffset, Impairment};
use sdrk_testkit::{Pcg32, VecStorage};

/// Samples per symbol and the symbol rate (the sample rate is `SPS` times it).
const SPS: usize = 4;
const SYMBOL_RATE: u64 = 10_000;
/// The carrier offset in cycles per symbol.
const OFFSET: f64 = 0.001;
/// Symbols per window of the slip count, symbols compared to find the alignment, and the lags
/// around the alignment a window is compared at.
const SLIP_WINDOW: usize = 2000;
const ALIGN_SYMBOLS: usize = 400;
const SLIP_LAGS: usize = 3;
/// The largest lag in symbols between the sent and the received symbols.
const MAX_LAG: usize = 80;

fn bps(order: u32) -> usize {
    if order == 64 { 6 } else { 4 }
}

/// `Q(x)`, the tail of the standard normal distribution.
fn q(x: f64) -> f64 {
    0.5 * libm::erfc(x / core::f64::consts::SQRT_2)
}

/// The Gray-coded approximation of M44 test 3 at `Es/N0` in dB:
/// `(4/k) (1 - 1/sqrt(M)) Q(sqrt(3/(M-1) Es/N0))`.
fn theory(order: u32, esn0_db: f64) -> f64 {
    let m = f64::from(order);
    let k = m.log2();
    let esn0 = 10f64.powf(esn0_db / 10.0);
    4.0 / k * (1.0 - 1.0 / m.sqrt()) * q((3.0 / (m - 1.0) * esn0).sqrt())
}

fn demod(order: u32, taps: usize) -> QamDemodBlock<VecStorage> {
    let cfg = QamDemodConfig {
        order,
        symbol_rate_hz: SYMBOL_RATE as f64,
        equalizer_taps: taps,
        ..QamDemodConfig::default()
    };
    QamDemodBlock::new(
        &cfg,
        Rate::hz(SYMBOL_RATE * SPS as u64).unwrap(),
        &mut VecStorage,
    )
    .unwrap()
}

/// The symbols (`k` bits each, the first the most significant) of the soft bits `soft`.
fn symbols(soft: &[f32], k: usize) -> Vec<u32> {
    soft.chunks_exact(k)
        .map(|b| b.iter().fold(0u32, |g, &s| (g << 1) | u32::from(s > 0.0)))
        .collect()
}

/// `n` random symbols of the seed through `qam_mod`, turned by `offset` (cycles per symbol), with Gaussian noise
/// at `esn0_db` (the measured mean power of the signal after the ramp of the filter); the
/// samples, the sent symbols and the power of the signal.
fn weak_signal(
    order: u32,
    n: usize,
    esn0_db: f64,
    offset: f64,
    seed: u64,
) -> (Vec<C32>, Vec<u32>, f64) {
    let k = bps(order);
    let mut rng = Pcg32::new(seed, 1);
    let bits: Vec<f32> = (0..(n + 64) * k)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect();
    let cfg = QamModConfig {
        order,
        samples_per_symbol: SPS,
        ..QamModConfig::default()
    };
    let rate = Rate::hz(SYMBOL_RATE * k as u64).unwrap();
    let mut tx = QamModBlock::<VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap();
    let mut sig = vec![C32::default(); n * SPS];
    assert_eq!(tx.process(&bits, &mut sig).unwrap().produced, sig.len());
    let wide: Vec<Complex64> = sig
        .iter()
        .map(|c| Complex64::new(f64::from(c.re), f64::from(c.im)))
        .collect();
    let fs = (SYMBOL_RATE as usize * SPS) as f64;
    let mut turned = Vec::with_capacity(wide.len());
    FrequencyOffset::new(offset * SYMBOL_RATE as f64, fs)
        .unwrap()
        .process(&wide, &mut turned);
    let body = &turned[100 * SPS..];
    let power = body.iter().map(Complex64::norm_sqr).sum::<f64>() / body.len() as f64;
    let ebn0 = esn0_db - 10.0 * (k as f64).log10();
    let mut awgn = Awgn::from_ebn0(ebn0, k as u32, SPS as f64, power, seed + 100).unwrap();
    let mut noisy = Vec::with_capacity(turned.len());
    awgn.process(&turned, &mut noisy);
    let input = noisy
        .iter()
        .map(|c| C32::new(c.re as f32, c.im as f32))
        .collect();
    (input, symbols(&bits, k), power)
}

/// Runs `input` through `block` in calls of 4096 samples; the soft bits.
fn demod_all(block: &mut QamDemodBlock<VecStorage>, input: &[C32]) -> Vec<f32> {
    let mut soft = Vec::new();
    let mut out = vec![0.0f32; 8192];
    for piece in input.chunks(4096) {
        let mut rest = piece;
        while !rest.is_empty() {
            let p = block.process(rest, &mut out).unwrap();
            soft.extend_from_slice(&out[..p.produced]);
            rest = &rest[p.consumed..];
        }
    }
    soft
}

/// The bit errors from the received symbol `from` on against `sent[j - lead - lag]` at the
/// alignment (rotation and lag) with the fewest bit errors over its first [`ALIGN_SYMBOLS`]
/// (the lag searched within `lead +- MAX_LAG` symbols of `lead`, the symbols of the lead-in):
/// the BER over the rest and the windows of [`SLIP_WINDOW`] symbols whose fewest bit errors
/// are at another rotation or at another lag within [`SLIP_LAGS`] (a lost or extra symbol or a
/// new rotation).
fn errors(order: u32, sent: &[u32], rx: &[u32], from: usize, lead: usize) -> (f64, u64) {
    let half = (bps(order) / 2) as u32;
    let back = |s: u32, r: u32| rotate_symbol(s, half, (4 - r) % 4);
    let err = |j: usize, r: u32, o: usize| {
        sent.get(j.wrapping_sub(o))
            .map_or(bps(order) as u32, |&t| (back(rx[j], r) ^ t).count_ones())
    };
    let mut best = (0u32, 0usize, u32::MAX);
    for r in 0..4 {
        for o in lead.saturating_sub(MAX_LAG)..lead + MAX_LAG {
            let e = (from..from + ALIGN_SYMBOLS).map(|j| err(j, r, o)).sum();
            if e < best.2 {
                best = (r, o, e);
            }
        }
    }
    let (r, o, _) = best;
    let start = from + ALIGN_SYMBOLS;
    let mut bits = 0u64;
    let mut wrong = 0u64;
    for j in start..rx.len() {
        bits += bps(order) as u64;
        wrong += u64::from(err(j, r, o));
    }
    let mut slips = 0u64;
    for w in (start..rx.len())
        .collect::<Vec<_>>()
        .chunks_exact(SLIP_WINDOW)
    {
        let count = |rr: u32, oo: usize| w.iter().map(|&j| err(j, rr, oo)).sum::<u32>();
        let here = count(r, o);
        let other = (0..4u32)
            .flat_map(|rr| (o.saturating_sub(SLIP_LAGS)..=o + SLIP_LAGS).map(move |oo| (rr, oo)))
            .filter(|&p| p != (r, o))
            .any(|(rr, oo)| count(rr, oo) < here);
        slips += u64::from(other);
    }
    (wrong as f64 / bits as f64, slips)
}

#[test]
fn a_correct_estimate_below_the_lock_level_is_kept() {
    let cases = [
        (16u32, 11usize, 11.5),
        (16, 11, 12.0),
        (16, 0, 12.0),
        (64, 11, 18.0),
        (64, 0, 18.0),
    ];
    let n = 100_000;
    let results: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .iter()
            .map(|&(order, taps, esn0)| {
                s.spawn(move || {
                    let (input, sent, _) = weak_signal(order, n, esn0, OFFSET, 780);
                    let mut block = demod(order, taps);
                    let soft = demod_all(&mut block, &input);
                    let rx = symbols(&soft, bps(order));
                    let (ber, slips) = errors(order, &sent, &rx, 5000, 0);
                    (block.restarts(), block.acquired_frequency(), ber, slips)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (&(order, taps, esn0), &(restarts, acquired, ber, slips)) in cases.iter().zip(&results) {
        let edge = theory(order, esn0 - 1.5);
        let what = format!("{order}-QAM, equalizer_taps {taps}, Es/N0 {esn0} dB");
        eprintln!(
            "{what}: restarts {restarts}, acquired {acquired:?}, BER {ber:.4e} (theory {:.4e}, \
             edge {edge:.4e}), slips {slips}",
            theory(order, esn0)
        );
        assert_eq!(restarts, 0, "{what}");
        let f = acquired.unwrap();
        assert!((f - OFFSET).abs() < 1e-4, "{what}: acquired {f}");
        assert_eq!(slips, 0, "{what}");
        assert!(ber <= edge, "{what}: BER {ber:.4e} above {edge:.4e}");
    }
}

#[test]
fn a_wrong_estimate_is_dropped_before_a_weak_signal() {
    let cases = [
        (16u32, 11usize, 12.0, true),
        (16, 11, 12.0, false),
        (16, 0, 12.0, true),
        (64, 11, 18.0, true),
        (64, 11, 18.0, false),
        (64, 0, 18.0, true),
    ];
    let n = 60_000;
    let results: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .iter()
            .map(|&(order, taps, esn0, noise_first)| {
                s.spawn(move || {
                    let (sig, sent, power) = weak_signal(order, n, esn0, OFFSET, 781);
                    // a carrier for 1000 symbols at the signal's amplitude (the first window
                    // takes its estimate), then the noise of the noise-first start
                    let amp = power.sqrt();
                    let mut input: Vec<C32> = (0..SPS * 1000)
                        .map(|j| {
                            let ph = core::f64::consts::TAU * 0.06 * j as f64 / SPS as f64;
                            C32::new((amp * ph.cos()) as f32, (amp * ph.sin()) as f32)
                        })
                        .collect();
                    if noise_first {
                        let sigma = (power / 2.0).sqrt();
                        let mut rng = Pcg32::new(781, 2);
                        input.extend((0..SPS * 20_480).map(|_| {
                            C32::new((sigma * rng.normal()) as f32, (sigma * rng.normal()) as f32)
                        }));
                    }
                    let lead = input.len() / SPS;
                    input.extend_from_slice(&sig);
                    let mut block = demod(order, taps);
                    let soft = demod_all(&mut block, &input);
                    let rx = symbols(&soft, bps(order));
                    let (ber, slips) = errors(order, &sent, &rx, rx.len() - 20_400, lead);
                    (block.restarts(), block.acquired_frequency(), ber, slips)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (&(order, taps, esn0, noise_first), &(restarts, acquired, ber, slips)) in
        cases.iter().zip(&results)
    {
        let edge = theory(order, esn0 - 1.5);
        let lead = if noise_first {
            "carrier, noise"
        } else {
            "carrier"
        };
        let what = format!("{order}-QAM, equalizer_taps {taps}, Es/N0 {esn0} dB after {lead}");
        eprintln!(
            "{what}: restarts {restarts}, acquired {acquired:?}, BER of the last 20000 symbols \
             {ber:.4e} (edge {edge:.4e}), slips {slips}"
        );
        assert!(restarts >= 1, "{what}");
        let f = acquired.unwrap();
        assert!((f - OFFSET).abs() < 1e-4, "{what}: acquired {f}");
        assert_eq!(slips, 0, "{what}");
        assert!(ber <= edge, "{what}: BER {ber:.4e} above {edge:.4e}");
    }
}

#[test]
fn an_estimate_at_the_edge_of_the_range_is_kept() {
    // F3 of the review of I44-9 (D1435): the fourth power cannot tell offsets 0.25 cycles per
    // symbol apart, so the estimate wraps into [-0.125, 0.125). For an offset just inside the
    // edge the estimate of a verify window can fall on either side while the loops hold the
    // first one; the verify step compares modulo a quarter cycle and keeps the estimate. A weak
    // 16-QAM signal (Es/N0 12 dB, below the lock level, so every check verifies) with the
    // equalizer, whose tracker holds the offset (`max_offset` of the carrier loop is 0.05).
    let n = 40_000;
    for (i, offset) in [0.124_97f64, -0.124_97].into_iter().enumerate() {
        let (input, _, _) = weak_signal(16, n, 12.0, offset, 782 + i as u64);
        let mut block = demod(16, 11);
        demod_all(&mut block, &input);
        let f = block.acquired_frequency().unwrap();
        let d = f - offset;
        let d = (d - 0.25 * (4.0 * d).round()).abs();
        eprintln!(
            "offset {offset}: restarts {}, acquired {f}, locked {}",
            block.restarts(),
            block.locked()
        );
        assert_eq!(block.restarts(), 0, "offset {offset}");
        assert!(d < 1e-4, "offset {offset}: acquired {f}");
    }
}
