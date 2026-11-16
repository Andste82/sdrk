//! M44.1 test 7 (the rule of D1707: no error state of the receiver lasts) and the step-2 part of
//! test 11: `qam_mod` -> carrier offset 0.001 cycles per symbol -> noise -> `qam_demod` with its
//! defaults (`equalizer_taps` 11, `carrier_bw` 0.0025) and without the equalizer, at the levels of
//! the BER tests of M44 (16-QAM at Eb/N0 14 dB, 64-QAM at Eb/N0 20 dB) and the gap case of
//! I44-17 (64-QAM at Es/N0 30 dB).
//!
//! - **Recovery** (seeds 844 to 848): the receiver is put into an error state and must return to
//!   one alignment (no lost symbol, no new rotation in any window of 2000 symbols) with the BER of
//!   the bound of M44 (the Gray approximation at Eb/N0 1 dB lower for 16-QAM, 1.5 dB for 64-QAM)
//!   from [`QAM_RECOVERY_SYMBOLS`] symbols after the error state on:
//!   - a gap of loud noise (at the signal's power, an AGC that turned the noise up) of 50000
//!     symbols, then the same signal on the same frequency (seed 844, and seed 848 for the gap
//!     case of I44-17 at 64-QAM 30 dB);
//!   - the weights of a false state (seed 845): in `Track` the equalizer's tracker is put off the
//!     carrier through the control interface (its frequency by 1e-4 cycles per symbol and its
//!     phase by 40 degrees), the mechanism of the false states of D1700, whose weights shrink;
//!   - a wrong phase at the hand-over (seed 846): 40 degrees added to the tracker's phase right
//!     after the equalizer's switch to the decision mode, the start that D1700 removed;
//!   - a step of the carrier frequency by 0.002 cycles per symbol (seed 847).
//!     The error states of the equalizer apply to the receivers with it.
//! - **Chunk invariance of step 2** (seed 851, test 11): two receivers whose controller takes the
//!   transitions of §6 (64-QAM at Es/N0 30 dB with the equalizer after the gap of seed 848,
//!   `Return`; 64-QAM with the equalizer in the injected false state of seed 845, `Shrink`) give
//!   the same soft bits in 20 random cut plans.
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
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use num_complex::Complex64;
use sdrk::Rate;
use sdrk_blocks::{Block, LockState, QamDemodBlock, QamDemodConfig, QamModBlock, QamModConfig};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::rotate_symbol;
use sdrk_core::kernels::equalize::EqMode;
use sdrk_testkit::channel::{Awgn, Impairment};
use sdrk_testkit::{Pcg32, VecStorage};

/// Samples per symbol and the symbol rate (the sample rate is `SPS` times it).
const SPS: usize = 4;
const SYMBOL_RATE: u64 = 10_000;
/// The carrier offset in cycles per symbol, and its step in the step case.
const OFFSET: f64 = 0.001;
const STEP: f64 = 0.002;
/// Symbols after the error state within which the receiver is back (one alignment and the BER of
/// the bound of M44 from there on): six checks of `qam_demod`. Measured (avx2) at most 20000
/// symbols, for 64-QAM with the equalizer after the injected false state (the weight watch acts
/// at about symbol 15900) and after the gap (the signal back, then the weight watch); every other
/// case within 8000 symbols. The margin of one check covers a weight watch that acts one check
/// later on another SIMD level (D1571).
const QAM_RECOVERY_SYMBOLS: usize = 24_576;
/// 40 degrees in units of `2^-32` turn of the equalizer's tracker.
const TURN_40: u32 = (40.0 / 360.0 * 4_294_967_296.0) as u32;
/// Symbols of the gap of loud noise.
const GAP: usize = 50_000;
/// Symbols before the error state (the receiver acquires and settles), and symbols counted after
/// the recovery.
const BEFORE: usize = 30_000;
const COUNTED: usize = 40_000;
/// Symbols per window of the slip count, symbols compared to find the alignment, the lags around
/// the alignment a window is compared at, and the largest lag after an error state.
const SLIP_WINDOW: usize = 2000;
const ALIGN_SYMBOLS: usize = 400;
const SLIP_LAGS: i64 = 3;
const MAX_LAG: i64 = 3000;

fn bps(order: u32) -> usize {
    if order == 64 { 6 } else { 4 }
}

/// `Q(x)`, the tail of the standard normal distribution.
fn q(x: f64) -> f64 {
    0.5 * libm::erfc(x / core::f64::consts::SQRT_2)
}

/// The Gray-coded approximation of M44 test 3 at `Es/N0` in dB.
fn theory(order: u32, esn0_db: f64) -> f64 {
    let m = f64::from(order);
    let k = libm::log2(m);
    let esn0 = libm::pow(10.0, esn0_db / 10.0);
    4.0 / k * (1.0 - 1.0 / libm::sqrt(m)) * q(libm::sqrt(3.0 / (m - 1.0) * esn0))
}

/// The bound of M44 at `Es/N0` in dB: the approximation 1 dB (16-QAM) or 1.5 dB (64-QAM) lower.
fn bound(order: u32, esn0_db: f64) -> f64 {
    theory(order, esn0_db - if order == 64 { 1.5 } else { 1.0 })
}

/// `Es/N0` in dB of `Eb/N0` in dB.
fn esn0_of(order: u32, ebn0_db: f64) -> f64 {
    ebn0_db + 10.0 * libm::log10(bps(order) as f64)
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

/// The error state of a case.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Event {
    /// A gap of loud noise of [`GAP`] symbols from symbol [`BEFORE`] on.
    Gap,
    /// The equalizer's tracker put off the carrier at symbol [`BEFORE`].
    Weights,
    /// A wrong phase of the tracker right after the switch to the decision mode.
    Phase,
    /// The carrier frequency steps by [`STEP`] at symbol [`BEFORE`].
    Step,
}

/// `n` random symbols of the seed through `qam_mod`, turned by the carrier (0.001 cycles per
/// symbol, plus [`STEP`] from symbol `step` on, the phase continuous; `libm` sines and cosines),
/// with Gaussian noise at `esn0_db` (the measured mean power of the signal after the ramp of the
/// filter), and a gap of noise at the signal's power over the symbols `gap`. The samples and the
/// sent symbols.
fn link(
    order: u32,
    n: usize,
    esn0_db: f64,
    step: Option<usize>,
    gap: Option<core::ops::Range<usize>>,
    seed: u64,
) -> (Vec<C32>, Vec<u32>) {
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
    let mut phase = 0.0f64;
    let turned: Vec<Complex64> = sig
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let f = OFFSET + step.map_or(0.0, |s| if i / SPS >= s { STEP } else { 0.0 });
            let r = Complex64::new(libm::cos(phase), libm::sin(phase));
            phase += core::f64::consts::TAU * f / SPS as f64;
            phase -= core::f64::consts::TAU * libm::floor(phase / core::f64::consts::TAU);
            Complex64::new(f64::from(c.re), f64::from(c.im)) * r
        })
        .collect();
    let body = &turned[100 * SPS..];
    let power = body.iter().map(Complex64::norm_sqr).sum::<f64>() / body.len() as f64;
    let ebn0 = esn0_db - 10.0 * libm::log10(k as f64);
    let mut awgn = Awgn::from_ebn0(ebn0, k as u32, SPS as f64, power, seed + 100).unwrap();
    let mut noisy = Vec::with_capacity(turned.len());
    awgn.process(&turned, &mut noisy);
    let mut input: Vec<C32> = noisy
        .iter()
        .map(|c| C32::new(c.re as f32, c.im as f32))
        .collect();
    if let Some(g) = gap {
        let sigma = libm::sqrt(power / 2.0);
        let mut rng = Pcg32::new(seed, 2);
        for c in &mut input[g.start * SPS..g.end * SPS] {
            *c = C32::new((sigma * rng.normal()) as f32, (sigma * rng.normal()) as f32);
        }
    }
    (input, symbols(&bits, k))
}

/// Runs `input` through `block` in calls of at most `piece` samples, the soft bits appended to
/// `soft`.
fn run(block: &mut QamDemodBlock<VecStorage>, input: &[C32], piece: usize, soft: &mut Vec<f32>) {
    let mut out = vec![0.0f32; 8192];
    for p in input.chunks(piece) {
        let mut rest = p;
        while !rest.is_empty() {
            let r = block.process(rest, &mut out).unwrap();
            soft.extend_from_slice(&out[..r.produced]);
            rest = &rest[r.consumed..];
        }
    }
}

/// Runs `input` through `block` in calls of 64 symbols and records the states of the controller
/// it passes through (the state and the symbol where it was first seen) after `soft`.
fn run_watched(
    block: &mut QamDemodBlock<VecStorage>,
    input: &[C32],
    soft: &mut Vec<f32>,
    path: &mut Vec<(LockState, usize)>,
) {
    let k = block.config().bits_per_symbol();
    for p in input.chunks(64 * SPS) {
        run(block, p, p.len(), soft);
        let state = block.watch().state();
        if path.last().is_none_or(|&(s, _)| s != state) {
            path.push((state, soft.len() / k));
        }
    }
}

/// The bit errors of the received symbol `j` turned back by `r` quarter turns against the sent
/// symbol `j - o` (all its bits when there is none).
fn err(order: u32, sent: &[u32], rx: &[u32], j: usize, r: u32, o: i64) -> u32 {
    let half = (bps(order) / 2) as u32;
    usize::try_from(j as i64 - o)
        .ok()
        .and_then(|t| sent.get(t))
        .map_or(bps(order) as u32, |&t| {
            (rotate_symbol(rx[j], half, (4 - r) % 4) ^ t).count_ones()
        })
}

/// The alignment (rotation and lag of the received against the sent symbols, the lag within
/// `lags`) with the fewest bit errors over the received symbols `from..from + len`, and those
/// errors.
fn align(
    order: u32,
    sent: &[u32],
    rx: &[u32],
    from: usize,
    len: usize,
    lags: core::ops::RangeInclusive<i64>,
) -> ((u32, i64), u32) {
    let mut best = ((0u32, 0i64), u32::MAX);
    for r in 0..4u32 {
        for o in lags.clone() {
            let mut e = 0u32;
            for j in from..from + len {
                e += err(order, sent, rx, j, r, o);
                if e >= best.1 {
                    break;
                }
            }
            if e < best.1 {
                best = ((r, o), e);
            }
        }
    }
    best
}

/// From the received symbol `from` on: the BER at the alignment of the first [`ALIGN_SYMBOLS`]
/// (any lag within [`MAX_LAG`]: the symbol clock runs free in a gap of the signal, and the
/// symbols lost or gained there are no error of the receiver), and the windows of
/// [`SLIP_WINDOW`] symbols whose fewest bit errors are at another rotation or at another lag
/// within [`SLIP_LAGS`].
fn errors(order: u32, sent: &[u32], rx: &[u32], from: usize) -> (f64, u64) {
    let ((r, o), _) = align(order, sent, rx, from, ALIGN_SYMBOLS, -MAX_LAG..=MAX_LAG);
    let start = from + ALIGN_SYMBOLS;
    let (mut bits, mut wrong) = (0u64, 0u64);
    for j in start..rx.len() {
        bits += bps(order) as u64;
        wrong += u64::from(err(order, sent, rx, j, r, o));
    }
    let mut slips = 0u64;
    for w in (start..rx.len())
        .collect::<Vec<_>>()
        .chunks_exact(SLIP_WINDOW)
    {
        let count = |rr: u32, oo: i64| {
            w.iter()
                .map(|&j| err(order, sent, rx, j, rr, oo))
                .sum::<u32>()
        };
        let here = count(r, o);
        let other = (0..4u32)
            .flat_map(|rr| (o - SLIP_LAGS..=o + SLIP_LAGS).map(move |oo| (rr, oo)))
            .filter(|&p| p != (r, o))
            .any(|(rr, oo)| count(rr, oo) < here);
        slips += u64::from(other);
    }
    (wrong as f64 / bits as f64, slips)
}

/// The fewest symbols after `event`, in steps of [`SLIP_WINDOW`], from which on the receiver
/// meets the condition of test 7 (one alignment, the BER at most `edge`): the measured recovery,
/// for the print and the log.
fn back_after(order: u32, sent: &[u32], rx: &[u32], event: usize, edge: f64) -> Option<usize> {
    (0..=QAM_RECOVERY_SYMBOLS + COUNTED / 2)
        .step_by(SLIP_WINDOW)
        .find(|&x| {
            let (ber, slips) = errors(order, sent, rx, event + x);
            slips == 0 && ber <= edge
        })
}

/// One receiver of test 7: the soft bits, the symbol of the error state, the transitions taken.
struct Outcome {
    what: String,
    order: u32,
    esn0: f64,
    event_at: usize,
    sent: Vec<u32>,
    soft: Vec<f32>,
    restarts: u32,
    locked: bool,
    acquired: Option<f64>,
    /// The states of the controller after the error state, with the symbol where each began.
    path: Vec<(LockState, usize)>,
}

/// Runs one case of test 7.
fn recover(order: u32, esn0: f64, taps: usize, event: Event, seed: u64) -> Outcome {
    let after = QAM_RECOVERY_SYMBOLS + COUNTED;
    let (n, step, gap) = match event {
        Event::Gap => (BEFORE + GAP + after, None, Some(BEFORE..BEFORE + GAP)),
        Event::Step => (BEFORE + after, Some(BEFORE), None),
        Event::Weights | Event::Phase => (BEFORE + after, None, None),
    };
    let (input, sent) = link(order, n, esn0, step, gap, seed);
    let mut block = demod(order, taps);
    let mut soft = Vec::new();
    let k = bps(order);
    let mut path = Vec::new();
    let event_at = match event {
        Event::Gap => {
            run(&mut block, &input[..(BEFORE + GAP) * SPS], 4096, &mut soft);
            run_watched(
                &mut block,
                &input[(BEFORE + GAP) * SPS..],
                &mut soft,
                &mut path,
            );
            BEFORE + GAP
        }
        Event::Step => {
            run(&mut block, &input[..BEFORE * SPS], 4096, &mut soft);
            run_watched(&mut block, &input[BEFORE * SPS..], &mut soft, &mut path);
            BEFORE
        }
        Event::Weights => {
            run(&mut block, &input[..BEFORE * SPS], 4096, &mut soft);
            let eq = block.equalizer_mut().unwrap();
            let (f, p) = (eq.state().frequency(), eq.state().phase());
            eq.set_frequency(f64::from(f) + 1e-4);
            eq.set_phase(p.wrapping_add(TURN_40));
            run_watched(&mut block, &input[BEFORE * SPS..], &mut soft, &mut path);
            BEFORE
        }
        Event::Phase => {
            // One symbol per call up to the switch to the decision mode.
            let mut at = 0;
            while block.equalizer().unwrap().mode() != EqMode::Decision {
                run(&mut block, &input[at..at + SPS], SPS, &mut soft);
                at += SPS;
            }
            let eq = block.equalizer_mut().unwrap();
            let p = eq.state().phase();
            eq.set_phase(p.wrapping_add(TURN_40));
            let event_at = soft.len() / k;
            run_watched(&mut block, &input[at..], &mut soft, &mut path);
            event_at
        }
    };
    Outcome {
        what: format!("{order}-QAM, Es/N0 {esn0:.2} dB, equalizer_taps {taps}, {event:?}"),
        order,
        esn0,
        event_at,
        sent,
        soft,
        restarts: block.restarts(),
        locked: block.locked(),
        acquired: block.acquired_frequency(),
        path,
    }
}

#[test]
fn every_error_state_returns_to_one_alignment() {
    let q16 = esn0_of(16, 14.0);
    let q64 = esn0_of(64, 20.0);
    let mut cases = Vec::new();
    for (order, esn0) in [(16u32, q16), (64, q64)] {
        for taps in [11usize, 0] {
            cases.push((order, esn0, taps, Event::Gap, 844));
            cases.push((order, esn0, taps, Event::Step, 847));
        }
        cases.push((order, esn0, 11, Event::Weights, 845));
        cases.push((order, esn0, 11, Event::Phase, 846));
    }
    for taps in [11usize, 0] {
        cases.push((64, 30.0, taps, Event::Gap, 848));
    }
    if let Ok(only) = std::env::var("ONLY") {
        cases.retain(|c| format!("{}-{}-{:?}", c.0, c.2, c.3) == only);
    }
    let results: Vec<Outcome> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .iter()
            .map(|&(order, esn0, taps, event, seed)| {
                s.spawn(move || recover(order, esn0, taps, event, seed))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut checked = Vec::new();
    for o in &results {
        let rx = symbols(&o.soft, bps(o.order));
        let edge = bound(o.order, o.esn0);
        let (ber, slips) = errors(o.order, &o.sent, &rx, o.event_at + QAM_RECOVERY_SYMBOLS);
        let back = back_after(o.order, &o.sent, &rx, o.event_at, edge);
        let path: Vec<String> = o
            .path
            .iter()
            .map(|(st, at)| format!("{st:?}@{}", *at as i64 - o.event_at as i64))
            .collect();
        eprintln!(
            "{}: restarts {}, locked {}, acquired {:?}, back {:?} symbols after the error \
             state, BER from {} symbols after it {ber:.4e} (bound {edge:.4e}), windows off the \
             alignment {slips}; the states after the error state {}",
            o.what,
            o.restarts,
            o.locked,
            o.acquired,
            back,
            QAM_RECOVERY_SYMBOLS,
            path.join(" ")
        );
        checked.push((o.what.clone(), ber, edge, slips));
    }
    for (what, ber, edge, slips) in checked {
        assert_eq!(slips, 0, "{what}: one alignment");
        assert!(ber <= edge, "{what}: BER {ber:.4e} above {edge:.4e}");
    }
}

/// Puts the receiver into the false state of seed 845 (test 7, `Event::Weights`): the tracker of
/// the equalizer 1e-4 cycles per symbol and 40 degrees off the carrier.
fn inject_false_weights(block: &mut QamDemodBlock<VecStorage>) {
    let eq = block.equalizer_mut().unwrap();
    let (f, p) = (eq.state().frequency(), eq.state().phase());
    eq.set_frequency(f64::from(f) + 1e-4);
    eq.set_phase(p.wrapping_add(TURN_40));
}

/// Runs `input` through `block` in the cut plan of `plans` (pieces of 1 to 9000 samples, room for
/// 1 to 16384 soft bits), the waiting soft bits flushed at the end; the soft bits appended to
/// `soft`.
fn run_plan(
    block: &mut QamDemodBlock<VecStorage>,
    input: &[C32],
    plans: &mut Pcg32,
    soft: &mut Vec<f32>,
) {
    let mut out = vec![0.0f32; 1 << 14];
    let mut at = 0;
    while at < input.len() {
        let len = 1 + plans.next_u32() as usize % 9000;
        let room = 1 + plans.next_u32() as usize % out.len();
        let end = (at + len).min(input.len());
        let r = block.process(&input[at..end], &mut out[..room]).unwrap();
        soft.extend_from_slice(&out[..r.produced]);
        at += r.consumed;
    }
    // the waiting soft bits
    loop {
        let r = block.process(&[], &mut out).unwrap();
        if r.produced == 0 {
            break;
        }
        soft.extend_from_slice(&out[..r.produced]);
    }
}

#[test]
fn the_recovery_is_chunk_invariant() {
    // (order, Es/N0, equalizer_taps, symbols, the gap, the injected false state, the seed of the
    // signal): the gap case of I44-17 (64-QAM at Es/N0 30 dB with the equalizer after the gap of
    // loud noise of seed 848, which only the signal back after the gap, `Return`, brings back, at
    // every SIMD level; the 16-QAM receiver without the equalizer of seed 844 takes `Return` at
    // `avx2` but not at the fallback, D1586), and 64-QAM with the equalizer put into the false
    // state of seed 845 at symbol `BEFORE` (the weight watch, `Shrink`); the cut plans of seed 851
    // (the input split at the injection)
    let gap = Some(BEFORE..BEFORE + GAP);
    let cases = [
        (
            64u32,
            30.0,
            11usize,
            BEFORE + GAP + QAM_RECOVERY_SYMBOLS + COUNTED,
            gap,
            false,
            848,
        ),
        (
            64,
            esn0_of(64, 20.0),
            11,
            BEFORE + QAM_RECOVERY_SYMBOLS,
            None,
            true,
            845,
        ),
    ];
    let mut plans = Pcg32::new(851, 3);
    for (order, esn0, taps, n, gap, inject, seed) in cases {
        let (input, _) = link(order, n, esn0, None, gap, seed);
        let split = if inject { BEFORE * SPS } else { input.len() };
        let run_whole = |recovery: bool| {
            let mut block = demod(order, taps);
            block.set_recovery(recovery);
            let mut soft = Vec::new();
            run(&mut block, &input[..split], split.max(1), &mut soft);
            if inject {
                inject_false_weights(&mut block);
                run(&mut block, &input[split..], input.len() - split, &mut soft);
            }
            (block, soft)
        };
        let (whole, want) = run_whole(true);
        let (step_one, _) = run_whole(false);
        let what = format!("{order}-QAM, Es/N0 {esn0:.2} dB, equalizer_taps {taps}");
        eprintln!(
            "{what}: restarts {} (without the recovery {})",
            whole.restarts(),
            step_one.restarts()
        );
        assert!(
            whole.restarts() > step_one.restarts(),
            "{what}: the recovery acts"
        );
        for _ in 0..20 {
            let mut block = demod(order, taps);
            let mut soft = Vec::new();
            run_plan(&mut block, &input[..split], &mut plans, &mut soft);
            if inject {
                inject_false_weights(&mut block);
                run_plan(&mut block, &input[split..], &mut plans, &mut soft);
            }
            assert_eq!(soft.len(), want.len(), "{what}");
            assert!(
                soft.iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{what}: the soft bits of a cut plan"
            );
            assert_eq!(block.restarts(), whole.restarts(), "{what}");
        }
    }
}
