//! The restart of the frequency acquisition of `qam_demod` below the lock level (I44-9, the
//! maintainer's decision D1604, implemented by D1605; decided by frequency since the maintainer's
//! decision D1618): `qam_mod` -> carrier offset 0.001 cycles
//! per symbol -> noise -> `qam_demod` with its defaults (`equalizer_taps` 11, `carrier_bw` 0.0025)
//! and without the equalizer.
//!
//! - **A correct estimate is kept** (seed 780): 16-QAM at Es/N0 11.5 and 12 dB and 64-QAM at
//!   18 dB, below the level at which the lock detector of `costas` turns on (16-QAM about
//!   14.8 dB, 64-QAM about 21.0 dB), over 100000 symbols (24 checks): the receiver never
//!   restarts, its estimate is within 1e-4 of the offset, the received symbols keep one
//!   alignment with the sent ones (every window of 2000 symbols has its fewest bit errors at the
//!   rotation and lag of the first 400 counted symbols, among the four rotations and that lag
//!   +-3 symbols: no lost symbol, no new rotation), and the BER is at most the Gray
//!   approximation of `qam_ber_m44` at Es/N0 1.5 dB lower. Before D1605 the check of D1421 restarted these receivers every 4864
//!   symbols (61 restarts in 300000 symbols, BER 0.12 to 0.24 measured per window of 2000
//!   symbols at the best alignment).
//! - **A wrong estimate is dropped** (seed 781): a carrier at 0.06 cycles per symbol (its
//!   fourth power is a clear peak, so the first estimate is the carrier's), then either the weak
//!   signal directly (the loops on a wrong frequency) or first Gaussian noise of the signal's
//!   power for 20480 symbols (the noise-first start, the loops on noise), then the weak signal:
//!   the receiver restarts at least once, ends with the estimate within 1e-4 of the offset, and
//!   the last 20000 symbols keep one alignment within the BER bound above (64-QAM with the
//!   equalizer after the carrier alone needs a second restart: its loops do not converge on the
//!   first estimate, and the verify window of D1623 finds them away from it).
//! - **A weak signal the loops hold is not restarted** (seeds 810 to 815, D1618): 16-QAM at
//!   Es/N0 9, 10 and 11 dB and 64-QAM at 15, 16 and 17 dB, with and without the equalizer, over
//!   200000 symbols (48 checks): the estimate is within 1e-4 of the offset. At these levels the lock level of a correctly derotated signal is that of noise,
//!   and the check of D1605 by the mean level restarted the receiver up to 34 times in 310000
//!   symbols; the check by frequency of D1618 restarts only on a clear verify window away from
//!   the loops. The receivers whose loops hold the signal with the loop constants of D1618 and
//!   D1622 (all without the equalizer, and with it 16-QAM at 11 dB and 64-QAM at 17 dB) keep one
//!   alignment over the run and never restart; the others (with the equalizer at 16-QAM 9 and
//!   10 dB and 64-QAM 15 and 16 dB) slip and may be restarted, which is printed (I44-17).
//! - **An estimate at the edge of the range is kept** (seeds 782 and 783, D1608): 16-QAM at
//!   Es/N0 12 dB with the equalizer at +-0.12497 cycles per symbol, where the estimate of a
//!   verify window can wrap to the other edge of `[-0.125, 0.125)`: no restart over 40000
//!   symbols, the estimate within 1e-4 modulo a quarter cycle per symbol.
//! - **A signal back after a gap is received again** (seed 816, F3 of the review of I44-13):
//!   16-QAM at Es/N0 24 dB and 64-QAM at 30 dB, with and without the equalizer, for 20000
//!   symbols, then a gap of five check intervals (20480 symbols) of Gaussian noise at the
//!   signal's noise floor or at its power (an AGC that turned the noise up), then the same
//!   signal on the same frequency for 40000 symbols: the estimate, kept over the gap by the
//!   check by frequency, is within 1e-4 of the offset, and from 16384 symbols after the gap the
//!   receiver is locked and keeps one alignment within the BER bound above. The equalizer of
//!   64-QAM at 30 dB adapted to the gap at the signal's power does not converge on the signal
//!   again and no check restarts it (the verify window agrees with the loops); that case is
//!   printed, not asserted (I44-17).
//! - **The decision mode starts at the phase of the output** (seed 817, I44-17): 64-QAM at
//!   Es/N0 18 and 19 dB, 0.0005 cycles per symbol and 20 ppm of clock drift (the signals of the
//!   comparison with GNU Radio of D1618), with 11 equalizer taps and with 15 taps and `sync_bw`
//!   0.002, over 60000 symbols: from symbol 10000 on one alignment, no restart, and the BER at
//!   most the Gray approximation at Es/N0 1 dB lower. Before the hand-over of the equalizer to the
//!   decision mode turned its tracker by the fourth-power phase of its output (D1700), all four
//!   receivers slipped (18 to 23 of the 25 windows off the first alignment, BER 0.38 to 0.44):
//!   the decision mode, started up to 40 degrees off, pulled the tracker's frequency into a false
//!   state that `costas` behind it hid from the verify window. And 16-QAM with 11 taps at Es/N0
//!   20 and 23 dB and 0.001 cycles per symbol without drift (the receivers of D1627 that
//!   converged after their CMA symbols on a false state, MSE 0.06 and the weights' energy 0.54):
//!   the same assertions.
//!
//! The soft bits of every receiver with the recovery of M44.1 §6 off (`set_recovery(false)`, the
//! receiver of step 1) keep the bits of the baseline of M44.1 (spec §5, test 5;
//! `sdrk_testkit::baseline`, table `qam_restart` of `tests/baseline/m44_1`), per SIMD level; the
//! assertions above hold for the receivers with the recovery on, as `qam_demod` is built (M44.1
//! test 8). The signals use the logarithms, powers, sines and cosines of `libm`, so that they are
//! the same bits on every host.
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
use sdrk_blocks::SymbolMapConstellation;
use sdrk_blocks::{Block, QamDemodBlock, QamDemodConfig, QamModBlock, QamModConfig};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::rotate_symbol;
use sdrk_testkit::baseline::{self, Baseline, Digest};
use sdrk_testkit::channel::{Awgn, ClockDrift, FrequencyOffset, Impairment};
use sdrk_testkit::{Pcg32, VecStorage};

/// The table of the baseline of M44.1 at the SIMD level of this build.
fn baseline_table() -> Baseline {
    Baseline::load(
        "m44_1",
        "qam_restart",
        baseline::level(sdrk_backend_simd::level_name()),
    )
    .unwrap()
}

/// Samples per symbol and the symbol rate (the sample rate is `SPS` times it).
const SPS: usize = 4;
const SYMBOL_RATE: u64 = 10_000;
/// The carrier offset in cycles per symbol.
const OFFSET: f64 = 0.001;
/// Symbols per window of the slip count, symbols compared to find the alignment, and the lags
/// around the alignment a window is compared at.
const SLIP_WINDOW: usize = 2000;
const ALIGN_SYMBOLS: usize = 400;
const SLIP_LAGS: i64 = 3;
/// Symbols of the gap of `a_signal_back_after_a_gap_is_received_again` (five check intervals of
/// `qam_demod`), and the symbols after its end within which the receiver takes the signal again.
const GAP_SYMBOLS: usize = 5 * 4096;
const BACK_WITHIN: usize = 4 * 4096;
/// The largest lag in symbols between the sent and the received symbols.
const MAX_LAG: i64 = 80;

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
        constellation: c_of(order),
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

/// Runs `input` through `block` with the recovery of M44.1 §6 off (the receiver of step 1,
/// `set_recovery(false)`) and checks its soft bits against the baseline of `case` (M44.1 test 5).
fn check_step_one(
    table: &Baseline,
    case: &str,
    mut block: QamDemodBlock<VecStorage>,
    input: &[C32],
) {
    block.set_recovery(false);
    let soft = demod_all(&mut block, input);
    table.check(case, &Digest::of_f32(&soft)).unwrap();
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
        constellation: c_of(order),
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
    let ebn0 = esn0_db - 10.0 * libm::log10(k as f64);
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
/// (the lag searched within `lead +- MAX_LAG` symbols of `lead`, the symbols of the lead-in; a
/// lag below `lead` is a receiver that lost symbols, for example in a gap of the signal):
/// the BER over the rest and the windows of [`SLIP_WINDOW`] symbols whose fewest bit errors
/// are at another rotation or at another lag within [`SLIP_LAGS`] (a lost or extra symbol or a
/// new rotation).
fn errors(order: u32, sent: &[u32], rx: &[u32], from: usize, lead: usize) -> (f64, u64) {
    let half = (bps(order) / 2) as u32;
    let back = |s: u32, r: u32| rotate_symbol(s, half, (4 - r) % 4);
    let err = |j: usize, r: u32, o: i64| {
        usize::try_from(j as i64 - o)
            .ok()
            .and_then(|t| sent.get(t))
            .map_or(bps(order) as u32, |&t| (back(rx[j], r) ^ t).count_ones())
    };
    let lead = lead as i64;
    let mut best = (0u32, 0i64, u32::MAX);
    for r in 0..4 {
        for o in lead - MAX_LAG..lead + MAX_LAG {
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
        let count = |rr: u32, oo: i64| w.iter().map(|&j| err(j, rr, oo)).sum::<u32>();
        let here = count(r, o);
        let other = (0..4u32)
            .flat_map(|rr| (o - SLIP_LAGS..=o + SLIP_LAGS).map(move |oo| (rr, oo)))
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
    let table = &baseline_table();
    let results: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .iter()
            .map(|&(order, taps, esn0)| {
                s.spawn(move || {
                    let (input, sent, _) = weak_signal(order, n, esn0, OFFSET, 780);
                    let mut block = demod(order, taps);
                    let soft = demod_all(&mut block, &input);
                    let case = format!("kept_qam{order}_taps{taps}_esn0_{esn0}");
                    check_step_one(table, &case, demod(order, taps), &input);
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
fn a_weak_signal_the_loops_hold_is_not_restarted() {
    // (order, Es/N0 in dB, seed, the equalizer_taps whose loops hold the signal with the loop
    // constants of D1618 and D1622: one alignment over the run); the others, with the equalizer
    // at the lower levels, slip, and the verify window may restart them (I44-17)
    let cases: [(u32, f64, u64, &[usize]); 6] = [
        (16, 9.0, 810, &[0]),
        (16, 10.0, 811, &[0]),
        (16, 11.0, 812, &[11, 0]),
        (64, 15.0, 813, &[0]),
        (64, 16.0, 814, &[0]),
        (64, 17.0, 815, &[11, 0]),
    ];
    let n = 200_000;
    let table = &baseline_table();
    let results: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .iter()
            .flat_map(|&(order, esn0, seed, holds)| {
                [11usize, 0].map(move |taps| (order, esn0, seed, taps, holds.contains(&taps)))
            })
            .map(|(order, esn0, seed, taps, holds)| {
                s.spawn(move || {
                    let (input, sent, _) = weak_signal(order, n, esn0, OFFSET, seed);
                    let mut block = demod(order, taps);
                    let soft = demod_all(&mut block, &input);
                    let case = format!("held_qam{order}_taps{taps}_esn0_{esn0}");
                    check_step_one(table, &case, demod(order, taps), &input);
                    let rx = symbols(&soft, bps(order));
                    let (ber, slips) = errors(order, &sent, &rx, 5000, 0);
                    let what = format!("{order}-QAM, equalizer_taps {taps}, Es/N0 {esn0} dB");
                    let state = (block.restarts(), block.acquired_frequency());
                    (what, holds, state, ber, slips)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (what, _, (restarts, acquired), ber, slips) in &results {
        eprintln!(
            "{what}: restarts {restarts}, acquired {acquired:?}, BER at the first alignment \
             {ber:.4e}, windows off it {slips}"
        );
    }
    for (what, holds, (restarts, acquired), _, slips) in &results {
        let f = acquired.unwrap();
        assert!((f - OFFSET).abs() < 1e-4, "{what}: acquired {f}");
        if *holds {
            assert_eq!(*slips, 0, "{what}: the loops hold the signal");
            assert_eq!(*restarts, 0, "{what}");
        }
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
    let table = &baseline_table();
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
                            C32::new((amp * libm::cos(ph)) as f32, (amp * libm::sin(ph)) as f32)
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
                    let first = if noise_first { "noise" } else { "carrier" };
                    let case = format!("dropped_qam{order}_taps{taps}_esn0_{esn0}_{first}");
                    check_step_one(table, &case, demod(order, taps), &input);
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
    // F3 of the review of I44-9 (D1608): the fourth power cannot tell offsets 0.25 cycles per
    // symbol apart, so the estimate wraps into [-0.125, 0.125). For an offset just inside the
    // edge the estimate of a verify window can fall on either side while the loops hold the
    // first one; the verify step compares modulo a quarter cycle and keeps the estimate. A weak
    // 16-QAM signal (Es/N0 12 dB, below the lock level, so every check verifies) with the
    // equalizer, whose tracker holds the offset (`max_offset` of the carrier loop is 0.05).
    let n = 40_000;
    let table = baseline_table();
    for (i, offset) in [0.124_97f64, -0.124_97].into_iter().enumerate() {
        let (input, _, _) = weak_signal(16, n, 12.0, offset, 782 + i as u64);
        let mut block = demod(16, 11);
        let soft = demod_all(&mut block, &input);
        table
            .check(&format!("edge_offset{offset}"), &Digest::of_f32(&soft))
            .unwrap();
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

#[test]
fn a_signal_back_after_a_gap_is_received_again() {
    // F3 of the review of I44-13: the check by frequency keeps the estimate over noise after a
    // transmission ends (the verify windows find no clear peak), and the equalizer goes on
    // adapting to the noise in the decision mode. The same signal on the same frequency after
    // the gap must be received again. (order, Es/N0 in dB, the gap at the signal's power (an
    // AGC that turned the noise up) or at its noise floor, the equalizer_taps whose receivers
    // take the signal again; the others are printed, I44-17)
    let cases: [(u32, f64, bool, &[usize]); 4] = [
        (16, 24.0, false, &[11, 0]),
        (16, 24.0, true, &[11, 0]),
        (64, 30.0, false, &[11, 0]),
        (64, 30.0, true, &[0]),
    ];
    let (before, gap, after) = (20_000usize, GAP_SYMBOLS, 40_000usize);
    let table = &baseline_table();
    let results: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .iter()
            .flat_map(|&(order, esn0, loud, holds)| {
                [11usize, 0].map(move |taps| (order, esn0, loud, taps, holds.contains(&taps)))
            })
            .map(|(order, esn0, loud, taps, holds)| {
                s.spawn(move || {
                    let n = before + gap + after;
                    let (mut input, sent, power) = weak_signal(order, n, esn0, OFFSET, 816);
                    // the noise of the signal (Es/N0 = SPS power / variance) or the signal power
                    let var = if loud {
                        power
                    } else {
                        power * SPS as f64 / libm::pow(10.0, esn0 / 10.0)
                    };
                    let sigma = (var / 2.0).sqrt();
                    let mut rng = Pcg32::new(816, 2);
                    for c in &mut input[before * SPS..(before + gap) * SPS] {
                        *c = C32::new((sigma * rng.normal()) as f32, (sigma * rng.normal()) as f32);
                    }
                    let mut block = demod(order, taps);
                    let mut soft = demod_all(&mut block, &input[..(before + gap) * SPS]);
                    let in_gap = block.restarts();
                    soft.extend(demod_all(&mut block, &input[(before + gap) * SPS..]));
                    let at = if loud { "loud" } else { "floor" };
                    let case = format!("gap_qam{order}_taps{taps}_esn0_{esn0}_{at}");
                    check_step_one(table, &case, demod(order, taps), &input);
                    let rx = symbols(&soft, bps(order));
                    let (ber, slips) = errors(order, &sent, &rx, before + gap + BACK_WITHIN, 0);
                    let what = format!(
                        "{order}-QAM, equalizer_taps {taps}, Es/N0 {esn0} dB, the gap at {}",
                        if loud {
                            "the signal power"
                        } else {
                            "the noise floor"
                        }
                    );
                    let state = (
                        in_gap,
                        block.restarts(),
                        block.locked(),
                        block.acquired_frequency(),
                    );
                    (what, esn0, order, holds, state, ber, slips)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (what, _, _, _, (in_gap, restarts, locked, acquired), ber, slips) in &results {
        eprintln!(
            "{what}: restarts {in_gap} up to the end of the gap, {restarts} at the end, locked \
             {locked}, acquired {acquired:?}, BER from {BACK_WITHIN} symbols after the gap \
             {ber:.4e}, windows off the alignment {slips}"
        );
    }
    for (what, esn0, order, holds, (_, _, locked, acquired), ber, slips) in &results {
        let f = acquired.unwrap();
        assert!((f - OFFSET).abs() < 1e-4, "{what}: acquired {f}");
        if *holds {
            let edge = theory(*order, esn0 - 1.5);
            assert!(*locked, "{what}: lock at the end");
            assert_eq!(*slips, 0, "{what}");
            assert!(*ber <= edge, "{what}: BER {ber:.4e} above {edge:.4e}");
        }
    }
}

/// `n` random symbols of the seed through `qam_mod`, turned by `offset` (cycles per symbol), its
/// clock stretched by `ppm`, with Gaussian noise at `esn0_db` (the measured mean power after the
/// ramp of the filter): the signals of the comparison of D1618. The samples and the sent symbols.
fn drifting_signal(
    order: u32,
    n: usize,
    esn0_db: f64,
    offset: f64,
    ppm: f64,
    seed: u64,
) -> (Vec<C32>, Vec<u32>) {
    let k = bps(order);
    let mut rng = Pcg32::new(seed, 1);
    let bits: Vec<f32> = (0..(n + 64) * k)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect();
    let cfg = QamModConfig {
        constellation: c_of(order),
        samples_per_symbol: SPS,
        ..QamModConfig::default()
    };
    let rate = Rate::hz(SYMBOL_RATE * k as u64).unwrap();
    let mut tx = QamModBlock::<VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap();
    let mut sig = vec![C32::default(); n * SPS];
    assert_eq!(tx.process(&bits, &mut sig).unwrap().produced, sig.len());
    let mut cur: Vec<Complex64> = sig
        .iter()
        .map(|c| Complex64::new(f64::from(c.re), f64::from(c.im)))
        .collect();
    let fs = (SYMBOL_RATE as usize * SPS) as f64;
    let mut stages: [Box<dyn Impairment>; 2] = [
        Box::new(FrequencyOffset::new(offset * SYMBOL_RATE as f64, fs).unwrap()),
        Box::new(ClockDrift::new(ppm).unwrap()),
    ];
    for s in &mut stages {
        let mut next = Vec::with_capacity(cur.len() + 64);
        s.process(&cur, &mut next);
        cur = next;
    }
    let body = &cur[100 * SPS..];
    let power = body.iter().map(Complex64::norm_sqr).sum::<f64>() / body.len() as f64;
    let ebn0 = esn0_db - 10.0 * libm::log10(k as f64);
    let mut awgn = Awgn::from_ebn0(ebn0, k as u32, SPS as f64, power, seed + 100).unwrap();
    let mut noisy = Vec::with_capacity(cur.len());
    awgn.process(&cur, &mut noisy);
    let input = noisy
        .iter()
        .map(|c| C32::new(c.re as f32, c.im as f32))
        .collect();
    (input, symbols(&bits, k))
}

#[test]
fn the_decision_mode_starts_at_the_phase_of_the_output() {
    // (order, Es/N0 in dB, equalizer_taps, sync_bw, offset in cycles per symbol, drift in ppm)
    let cases = [
        (64u32, 18.0, 11usize, 0.005, 0.0005, 20.0),
        (64, 18.0, 15, 0.002, 0.0005, 20.0),
        (64, 19.0, 11, 0.005, 0.0005, 20.0),
        (64, 19.0, 15, 0.002, 0.0005, 20.0),
        (16, 20.0, 11, 0.005, OFFSET, 0.0),
        (16, 23.0, 11, 0.005, OFFSET, 0.0),
    ];
    let n = 60_000;
    let table = &baseline_table();
    let results: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .iter()
            .map(|&(order, esn0, taps, sync_bw, offset, ppm)| {
                s.spawn(move || {
                    let (input, sent) = drifting_signal(order, n, esn0, offset, ppm, 817);
                    let cfg = QamDemodConfig {
                        constellation: c_of(order),
                        symbol_rate_hz: SYMBOL_RATE as f64,
                        equalizer_taps: taps,
                        sync_bw: Some(sync_bw),
                        ..QamDemodConfig::default()
                    };
                    let rate = Rate::hz(SYMBOL_RATE * SPS as u64).unwrap();
                    let mut block = QamDemodBlock::new(&cfg, rate, &mut VecStorage).unwrap();
                    let soft = demod_all(&mut block, &input);
                    let case = format!(
                        "handover_qam{order}_taps{taps}_sync{sync_bw}_esn0_{esn0}_offset{offset}"
                    );
                    check_step_one(
                        table,
                        &case,
                        QamDemodBlock::new(&cfg, rate, &mut VecStorage).unwrap(),
                        &input,
                    );
                    let rx = symbols(&soft, bps(order));
                    let (ber, slips) = errors(order, &sent, &rx, 10_000, 0);
                    let what = format!(
                        "{order}-QAM, equalizer_taps {taps}, sync_bw {sync_bw}, Es/N0 {esn0} dB"
                    );
                    (what, order, esn0, block.restarts(), ber, slips)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (what, order, esn0, restarts, ber, slips) in &results {
        eprintln!(
            "{what}: restarts {restarts}, BER at the first alignment {ber:.4e} (bound {:.4e}), \
             windows off it {slips}",
            theory(*order, esn0 - 1.0)
        );
    }
    for (what, order, esn0, restarts, ber, slips) in &results {
        assert_eq!(*slips, 0, "{what}: one alignment from symbol 10000 on");
        assert_eq!(*restarts, 0, "{what}");
        assert!(*ber <= theory(*order, esn0 - 1.0), "{what}: BER {ber}");
    }
}

/// The constellation of a numeric order of the tables of this file (2, 4, 8, 16 or 64).
fn c_of(order: u32) -> SymbolMapConstellation {
    match order {
        2 => SymbolMapConstellation::Bpsk,
        4 => SymbolMapConstellation::Qpsk,
        8 => SymbolMapConstellation::Psk8,
        16 => SymbolMapConstellation::Qam16,
        64 => SymbolMapConstellation::Qam64,
        other => unreachable!("no constellation of the order {other}"),
    }
}
