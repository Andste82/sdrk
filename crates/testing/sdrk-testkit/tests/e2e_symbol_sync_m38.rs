//! M38 tests 2 and 3: the lock of the Gardner symbol synchronizer (seed 494) and the bit error
//! rate of BPSK through `symbol_sync` -> `slice` -> `BerMeter` (Gardner, seed 495;
//! Mueller–Müller, seed 496). The measured values are printed (run with `--nocapture` to see
//! them).
//!
//! - **Lock:** a raised-cosine BPSK signal (roll-off 0.35, no noise) whose first symbol lies
//!   half a symbol after the first strobe of the synchronizer, with a clock drift of +-100 ppm,
//!   at 2, 4, 4.1 and 10.1 samples per symbol. The kernel runs one input sample per call, so
//!   every strobe position is known (`SymbolSyncState::next_strobe`, D556); from the strobe
//!   500 on, every strobe lies within 0.05 symbols of the nearest symbol centre of the
//!   generator (moved by the drift with `ClockDrift::output_position`).
//! - **BER:** PRBS-15 BPSK with a root-raised-cosine pulse (roll-off 0.35) and the matched
//!   root-raised-cosine filter in the test, so that the whole chain has the raised-cosine
//!   response 0.35 (D557); 50 ppm clock drift, then white Gaussian noise of the Eb/N0 point
//!   (`Awgn::from_ebn0` with the measured mean power of the transmitted signal). The matched
//!   filter is scaled by `1 / sum h^2`, so the symbols have the amplitude 1 at the strobes and
//!   the noise at the decision has `sigma^2 = 1 / (2 Eb/N0)`: the theory is `Q(sqrt(2 Eb/N0))`.
//!   Each Eb/N0 point is one receiver on the same transmitted and drifted signal. The first
//!   `SETTLE_BITS` soft bits are skipped (lock); then exactly 10^6 bits are counted after the
//!   meter's synchronization.
//!
//! The file needs the feature `std` (for the channel models and `VecStorage`).

#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::print_stderr,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use num_complex::Complex64;
use sdrk::{Rate, Sdrk};
use sdrk_blocks::{
    Block, SliceBlock, SliceConfig, SliceConstellation, SymbolSyncBlock, SymbolSyncConfig, Ted,
};
use sdrk_core::C32;
use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncState};
use sdrk_proto::prbs::{Poly, Prbs};
use sdrk_testkit::VecStorage;
use sdrk_testkit::ber::BerMeter;
use sdrk_testkit::channel::{Awgn, ClockDrift, Impairment};
use sdrk_testkit::r#gen::{Psk, PskTest, Pulse, pulse};
use sdrk_testkit::rng::Pcg32;

// ---- test 2: lock ------------------------------------------------------------------------

/// Seed of the lock test (random bits; stream = the case number).
const LOCK_SEED: u64 = 494;
/// Strobes before the timing error is checked.
const LOCK_AFTER: usize = 500;
/// Largest timing error after `LOCK_AFTER` strobes, in symbols.
const LOCK_MAX_ERROR: f64 = 0.05;
/// Symbols of the lock signal (the error is checked over the last 1000 of 1500 strobes).
const LOCK_SYMBOLS: usize = 1_600;
/// Strobes checked.
const LOCK_STROBES: usize = 1_500;
/// Roll-off of the test signals.
const ROLLOFF: f64 = 0.35;

/// The slope of the detector `ted` at the timing offset 0 for random `+-1` symbols with the
/// raised-cosine pulse `p` (roll-off 0.35, the whole chain), per symbol of offset `tau`:
/// `dE[e]/dtau` at 0 with, for Gardner, `E[e](tau) = sum_n (p(n - 1 + tau) - p(n + tau))
/// p(n - 1/2 + tau)` and, for Mueller–Müller with correct decisions,
/// `E[e](tau) = p(1 + tau) - p(tau - 1)`. Both are negative: a late strobe gives a negative
/// error. About -1.078 (Gardner) and -1.780 (Mueller–Müller).
fn slope_per_symbol(ted: Ted) -> f64 {
    let p = |t: f64| pulse(Pulse::RaisedCosine, ROLLOFF, t);
    let mean = |tau: f64| -> f64 {
        if ted == Ted::Gardner {
            (-64i32..=64)
                .map(|n| {
                    let n = f64::from(n);
                    (p(n - 1.0 + tau) - p(n + tau)) * p(n - 0.5 + tau)
                })
                .sum()
        } else {
            p(1.0 + tau) - p(tau - 1.0)
        }
    };
    let h = 1e-4;
    (mean(h) - mean(-h)) / (2.0 * h)
}

/// The block default `ted_gain` 1.0 is the slope of the Gardner detector per symbol for this
/// signal within 10 % (D565): the loop of the defaults has about its nominal bandwidth at every
/// `sps`. Mueller–Müller is steeper (about 1.78), so its loop is somewhat wider.
#[test]
fn the_default_detector_gain_is_the_gardner_slope_per_symbol() {
    let gardner = slope_per_symbol(Ted::Gardner).abs();
    let mm = slope_per_symbol(Ted::MuellerMuller).abs();
    eprintln!("detector slope per symbol: Gardner {gardner:.3}, Mueller-Mueller {mm:.3}");
    let default = SymbolSyncConfig::default().ted_gain;
    assert!(
        (gardner / default - 1.0).abs() < 0.1,
        "Gardner slope {gardner}"
    );
    assert!(mm > gardner);
}

/// The loop of the block defaults (§2: `loop_bw` 0.01, `damping` 0.7071, `max_deviation`
/// 0.005, `ted_gain` 1.0).
fn lock_state(sps: f64) -> SymbolSyncState<C32> {
    let c = SymbolSyncConfig::default();
    SymbolSyncState::new(
        sps,
        Ted::Gardner,
        c.loop_bw,
        c.damping,
        c.max_deviation,
        c.ted_gain,
    )
    .unwrap()
}

/// Converts to `C32` (rounding each component once).
fn to_c32(x: &[Complex64]) -> Vec<C32> {
    x.iter()
        .map(|v| C32::new(v.re as f32, v.im as f32))
        .collect()
}

/// Returns the largest timing error (in symbols) of the strobes `LOCK_AFTER..LOCK_STROBES`
/// and the error of every strobe.
fn lock_run(case: u64, sps: f64, ppm: f64) -> (f64, Vec<f64>) {
    let mut rng = Pcg32::new(LOCK_SEED, case);
    let bits: Vec<u8> = (0..LOCK_SYMBOLS)
        .map(|_| (rng.next_u32() & 1) as u8)
        .collect();
    let mut g = PskTest::new(Psk::Bpsk, sps, ROLLOFF);
    // The first strobe is at input sample 1 (D530): the first symbol lies half a symbol later.
    g.offset = 1.0 + 0.5 * sps;
    let mut tx = vec![Complex64::new(0.0, 0.0); g.samples(&bits)];
    g.generate(&bits, 0, &mut tx);
    let mut drift = ClockDrift::new(ppm).unwrap();
    let mut rx = Vec::new();
    drift.process(&tx, &mut rx);
    let rx = to_c32(&rx);

    let mut st = lock_state(sps);
    let mut out = [C32::new(0.0, 0.0); 2];
    let mut consumed = 0usize;
    let mut strobes = Vec::new();
    for x in &rx {
        let at = consumed as f64 + st.next_strobe();
        let p = Sdrk
            .symbol_sync(&mut st, core::slice::from_ref(x), &mut out)
            .unwrap();
        assert!(p.produced <= 1, "one sample yields at most one strobe");
        if p.produced == 1 {
            strobes.push(at);
        }
        consumed += p.consumed;
        if strobes.len() == LOCK_STROBES {
            break;
        }
    }
    assert_eq!(strobes.len(), LOCK_STROBES, "too few strobes");

    // The symbol k is at the generator position offset + k sps, which the drift moves to
    // output_position(...); one symbol is sps / ratio received samples.
    let period = drift.output_position(sps);
    let errors: Vec<f64> = strobes
        .iter()
        .map(|&s| {
            let k = ((s * drift.ratio() - g.offset) / sps).round().max(0.0) as usize;
            (s - drift.output_position(g.symbol_position(k))) / period
        })
        .collect();
    let max = errors[LOCK_AFTER..]
        .iter()
        .fold(0.0f64, |m, e| m.max(e.abs()));
    (max, errors)
}

/// The strobe after which the error stays below `LOCK_MAX_ERROR` (for the report).
fn settled_at(errors: &[f64]) -> usize {
    errors
        .iter()
        .rposition(|e| e.abs() >= LOCK_MAX_ERROR)
        .map_or(0, |i| i + 1)
}

#[test]
fn gardner_locks_within_500_symbols() {
    let mut failures = Vec::new();
    let mut case = 0u64;
    for sps in [2.0, 4.0, 4.1, 10.1] {
        for ppm in [100.0, -100.0] {
            case += 1;
            let (max, errors) = lock_run(case, sps, ppm);
            let rms_tail = (errors[LOCK_AFTER..].iter().map(|e| e * e).sum::<f64>()
                / (LOCK_STROBES - LOCK_AFTER) as f64)
                .sqrt();
            eprintln!(
                "lock sps {sps:5.1} drift {ppm:+5.0} ppm: start error {:+.3} symbols, \
                 below {LOCK_MAX_ERROR} from strobe {}, max error after {LOCK_AFTER}: {max:.4} \
                 symbols, rms {rms_tail:.4}",
                errors[0],
                settled_at(&errors)
            );
            if max >= LOCK_MAX_ERROR {
                failures.push(format!("sps {sps} drift {ppm} ppm: max error {max:.4}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

// ---- test 3: BER ------------------------------------------------------------------------

/// Bits counted per Eb/N0 point.
const BER_BITS: u64 = 1_000_000;
/// Soft bits skipped before the meter (lock of the synchronizer).
const SETTLE_BITS: usize = 1_000;
/// Drift of the BER test in ppm.
const BER_DRIFT_PPM: f64 = 50.0;
/// Pulse span of the transmitter and the matched filter in symbols per side.
const SPAN: usize = 16;
/// Samples generated per step.
const CHUNK: usize = 1 << 15;
/// Symbol rate of the BER test in Hz (the sample rate is `sps` times this).
const SYMBOL_RATE: u64 = 10_000;
/// Initial register of the PRBS and the meters (all ones).
const PRBS_SEED: u32 = 0x7fff;

/// `Q(x)`, the tail of the standard normal distribution.
fn q(x: f64) -> f64 {
    0.5 * libm::erfc(x / core::f64::consts::SQRT_2)
}

/// The BPSK theory `Q(sqrt(2 Eb/N0))`.
fn bpsk_theory(ebn0_db: f64) -> f64 {
    q((2.0 * 10f64.powf(ebn0_db / 10.0)).sqrt())
}

/// The matched root-raised-cosine filter `h[n] = rrc(n / sps)`, `|n| <= SPAN sps`, scaled by
/// `1 / sum h^2` (unit gain for the symbol amplitude), with its state.
struct MatchedFilter {
    taps: Vec<f64>,
    hist: Vec<Complex64>,
}

impl MatchedFilter {
    fn new(sps: f64) -> Self {
        let half = (SPAN as f64 * sps).floor() as i64;
        let raw: Vec<f64> = (-half..=half)
            .map(|n| pulse(Pulse::RootRaisedCosine, ROLLOFF, n as f64 / sps))
            .collect();
        let energy: f64 = raw.iter().map(|h| h * h).sum();
        let taps: Vec<f64> = raw.iter().map(|h| h / energy).collect();
        let hist = vec![Complex64::new(0.0, 0.0); taps.len() - 1];
        Self { taps, hist }
    }

    /// Filters `x` (one output per input; the filter is symmetric, so no reversal is needed).
    fn process(&mut self, x: &[Complex64]) -> Vec<Complex64> {
        let l = self.taps.len();
        self.hist.extend_from_slice(x);
        let out = self
            .hist
            .windows(l)
            .map(|w| {
                let mut re = 0.0;
                let mut im = 0.0;
                for (v, h) in w.iter().zip(&self.taps) {
                    re += v.re * h;
                    im += v.im * h;
                }
                Complex64::new(re, im)
            })
            .collect();
        let keep = self.hist.len() - (l - 1);
        self.hist.drain(..keep);
        out
    }
}

/// One receiver: noise, matched filter, `symbol_sync`, `slice bpsk`, `BerMeter`.
struct Receiver {
    ebn0_db: f64,
    awgn: Awgn,
    mf: MatchedFilter,
    sync: SymbolSyncBlock<C32, VecStorage>,
    slice: SliceBlock<C32, VecStorage>,
    meter: BerMeter,
    skipped: usize,
    symbols: Vec<C32>,
    soft: Vec<f32>,
}

impl Receiver {
    fn new(ebn0_db: f64, sps: f64, ted: Ted, power: f64, seed: u64) -> Self {
        let fs = Rate::hz((sps * SYMBOL_RATE as f64).round() as u64).unwrap();
        let mut storage = VecStorage;
        // Gardner runs with the block default `ted_gain` 1.0 (its slope is about 1.08);
        // Mueller–Müller with its own slope per symbol (about 1.78), as `docs/blocks.md` advises:
        // with 1.0 its loop is 1.78 times wider and slips at 6 dB (D565).
        let defaults = SymbolSyncConfig::default();
        let ted_gain = match ted {
            Ted::MuellerMuller => slope_per_symbol(ted).abs(),
            _ => defaults.ted_gain,
        };
        let sync_cfg = SymbolSyncConfig {
            symbol_rate_hz: SYMBOL_RATE as f64,
            ted,
            ted_gain,
            ..defaults
        };
        let sync = SymbolSyncBlock::new(&sync_cfg, fs, &mut storage).unwrap();
        assert_eq!(sync.sps_estimate(), sps);
        let slice_cfg = SliceConfig {
            constellation: SliceConstellation::Bpsk,
            scale: 1.0,
        };
        let slice = SliceBlock::new(&slice_cfg, fs, &mut storage).unwrap();
        Self {
            ebn0_db,
            awgn: Awgn::from_ebn0(ebn0_db, 1, sps, power, seed).unwrap(),
            mf: MatchedFilter::new(sps),
            sync,
            slice,
            meter: BerMeter::new(Poly::Prbs15, PRBS_SEED).unwrap(),
            skipped: 0,
            symbols: vec![C32::new(0.0, 0.0); CHUNK],
            soft: vec![0.0; CHUNK],
        }
    }

    fn done(&self) -> bool {
        self.meter.bits() >= BER_BITS
    }

    /// Takes clean received samples (after the drift).
    fn process(&mut self, clean: &[Complex64]) {
        if self.done() {
            return;
        }
        let mut noisy = Vec::with_capacity(clean.len());
        self.awgn.process(clean, &mut noisy);
        let filtered = to_c32(&self.mf.process(&noisy));
        let mut rest = &filtered[..];
        loop {
            let p = self.sync.process(rest, &mut self.symbols).unwrap();
            rest = &rest[p.consumed..];
            let s = self
                .slice
                .process(&self.symbols[..p.produced], &mut self.soft)
                .unwrap();
            assert_eq!((s.consumed, s.produced), (p.produced, p.produced));
            for &b in &self.soft[..s.produced] {
                if self.skipped < SETTLE_BITS {
                    self.skipped += 1;
                } else if !self.done() {
                    self.meter.push(u8::from(b > 0.0));
                }
            }
            if rest.is_empty() && p.produced < self.symbols.len() {
                break;
            }
        }
    }
}

/// `(Eb/N0 in dB, measured BER, theory, 95 % interval, slips)` of one Eb/N0 point.
type BerPoint = (f64, f64, f64, (f64, f64), u64);

/// Runs PRBS-15 BPSK at `sps` with `ted` through all `points` (Eb/N0 in dB).
fn ber_run(sps: f64, ted: Ted, seed: u64, points: &[f64]) -> Vec<BerPoint> {
    let mut prbs = Prbs::new(Poly::Prbs15, PRBS_SEED).unwrap();
    // Enough symbols for the settling, the filter delays and 10^6 counted bits.
    let nsym = BER_BITS as usize + SETTLE_BITS + 4 * SPAN + 1_000;
    let bits: Vec<u8> = (0..nsym).map(|_| prbs.next_bit()).collect();
    let mut g = PskTest::new(Psk::Bpsk, sps, ROLLOFF);
    g.pulse = Pulse::RootRaisedCosine;
    g.span = SPAN;
    let total = g.samples(&bits);

    // The mean power of the transmitted signal per sample (the pulse energy of one symbol is
    // about sps samples, so this is close to 1), measured over the first 2^16 samples after
    // the ramp-up of the first pulse.
    let mut probe = vec![Complex64::new(0.0, 0.0); 1 << 16];
    let skip = (2.0 * SPAN as f64 * sps) as u64;
    g.generate(&bits, skip, &mut probe);
    let power = probe.iter().map(Complex64::norm_sqr).sum::<f64>() / probe.len() as f64;
    eprintln!("ber sps {sps}: mean signal power {power:.5} per sample");

    let mut rxs: Vec<Receiver> = points
        .iter()
        .map(|&e| Receiver::new(e, sps, ted, power, seed))
        .collect();
    let mut drift = ClockDrift::new(BER_DRIFT_PPM).unwrap();
    let mut tx = vec![Complex64::new(0.0, 0.0); CHUNK];
    let mut clean = Vec::with_capacity(CHUNK);
    let mut start = 0usize;
    while start < total && !rxs.iter().all(Receiver::done) {
        let n = CHUNK.min(total - start);
        g.generate(&bits, start as u64, &mut tx[..n]);
        clean.clear();
        drift.process(&tx[..n], &mut clean);
        for rx in &mut rxs {
            rx.process(&clean);
        }
        start += n;
    }
    rxs.iter()
        .map(|rx| {
            assert!(
                rx.done(),
                "Eb/N0 {}: only {} bits counted",
                rx.ebn0_db,
                rx.meter.bits()
            );
            (
                rx.ebn0_db,
                rx.meter.ber(),
                bpsk_theory(rx.ebn0_db),
                rx.meter.confidence_95(),
                rx.meter.slips(),
            )
        })
        .collect()
}

/// Checks every point against the theory within `factor` (both ways) and prints it.
fn check_ber(name: &str, results: &[BerPoint], factor: f64) {
    let mut failures = Vec::new();
    for &(ebn0, ber, theory, (lo, hi), slips) in results {
        eprintln!(
            "{name} Eb/N0 {ebn0} dB: BER {ber:.4e} (95 % interval {lo:.4e} .. {hi:.4e}), \
             theory {theory:.4e}, ratio {:.3}, slips {slips}",
            ber / theory
        );
        if !(ber <= theory * factor && ber >= theory / factor) || slips != 0 {
            failures.push(format!(
                "{name} Eb/N0 {ebn0} dB: BER {ber:.4e} vs theory {theory:.4e}, slips {slips}"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
fn gardner_bpsk_ber_within_1_5_of_theory() {
    let results = ber_run(4.1, Ted::Gardner, 495, &[6.0, 8.0]);
    check_ber("gardner sps 4.1", &results, 1.5);
}

#[test]
fn mueller_muller_bpsk_ber_within_2_of_theory() {
    let results = ber_run(2.0, Ted::MuellerMuller, 496, &[6.0, 8.0]);
    check_ber("mueller-muller sps 2", &results, 2.0);
}
