//! M44.2 tests 3 to 5: `qam_mod` -> channel -> noise -> `qam_demod` for the square orders of
//! M44.2 (`qpsk`, `qam256`, `qam1024`; 16-QAM and 64-QAM keep the tests of M44 in
//! `qam_ber_m44`).
//!
//! - **Test 3, BER over AWGN** against the Gray-coded approximation
//!   `P_b = (4/k) (1 - 1/sqrt(M)) Q(sqrt(3k/(M-1) Eb/N0))` over 10^6 bits: 4-QAM (`qpsk`) at
//!   Eb/N0 8 and 10 dB within 1 dB (seed 825), 256-QAM at 22 and 24 dB within 1.5 dB (seed 826),
//!   1024-QAM at 27 and 29 dB within 2 dB (seed 827). The carrier loop reports lock at the first
//!   counted symbol (test 6 at the block: the lock detector is on at the SNRs of test 3).
//! - **Test 4, multipath** (seed 828): 256-QAM through `Multipath::mild()` with `equalizer_taps`
//!   15 within 3 dB of the AWGN formula at Eb/N0 24 dB; without the equalizer the BER is above
//!   1e-2.
//! - **Test 5, offset and drift**: 256-QAM with 0.001 cycles per symbol and 20 ppm at Eb/N0
//!   24 dB (seed 829) acquires within 5000 symbols without the equalizer (whose blind start of
//!   256-QAM is longer, D1517), 1024-QAM with 0.0005 cycles per symbol and 20 ppm at 29 dB (seed
//!   830) with the equalizer within [`QAM1024_ACQUISITION_SYMBOLS`] (without it the residual
//!   interference of the pulse leaves 1024-QAM at the edge of the 2 dB window, D1522): lock at
//!   that symbol, and from there the BER of test 3 without slips; the frequency acquisition
//!   finds the offset within 1e-4 cycles per symbol; the
//!   noise-first start of 256-QAM (seed 831): a carrier at 0.06 cycles per symbol for 1000
//!   symbols (its estimate is wrong for the signal), Gaussian noise of the signal's power for
//!   20480 symbols, then the signal at 0.001 cycles per symbol: the receiver restarts, ends with
//!   the estimate within 1e-4 and keeps one alignment over its last 20000 symbols within the BER
//!   of test 3.
//!
//! "Within x dB" (as M44, D818 of M43): the measured BER is at most the formula at
//! `Eb/N0 - x dB` over the counted bits, with no slip of the meter.
//!
//! The chain is that of `qam_ber_m44` (D1217): random bits (stream 1 of the `Pcg32` of the
//! test's seed) as soft bits `+-1` -> `QamModBlock` (4 samples per symbol, roll-off 0.35, span
//! 10) -> the channel -> `Awgn::from_ebn0` (the test's seed) with the measured mean power of the
//! signal after the channel -> `QamDemodBlock` with its defaults (among them the `sync_bw` of the
//! constellation, 0.002 for 256-QAM and 1024-QAM, D1527) and the `equalizer_taps` of the test.
//! The first symbols
//! are skipped (the training of the equalizer, [`training_symbols`], or the acquisition); the
//! next 400 received symbols fix the alignment with the sent symbols (the lag and the rotation
//! with the fewest bit errors), and the bit errors are counted over the next 10^6 bits. A window
//! of 2000 symbols with more than 5 % symbol errors counts as a slip.
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
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use num_complex::Complex64;
use sdrk::Rate;
use sdrk_blocks::SymbolMapConstellation;
use sdrk_blocks::{Block, QamDemodBlock, QamDemodConfig, QamModBlock, QamModConfig};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::rotate_symbol;
use sdrk_testkit::channel::{Awgn, ClockDrift, FrequencyOffset, Impairment, Multipath};
use sdrk_testkit::{Pcg32, VecStorage};

/// Bits counted per receiver.
const BER_BITS: u64 = 1_000_000;
/// Symbols skipped before counting in tests 3 and 4 (the training of the equalizer: its blind
/// symbols, the decision mode and the first check of the controller, D1517): 3000 for 4-QAM
/// (1000 blind symbols, as M44), 20000 for 256-QAM (8000, the check at 12288), 35000 for
/// 1024-QAM (16000, the check at 20480).
fn training_symbols(order: u32) -> usize {
    match order {
        256 => 20_000,
        1024 => 35_000,
        _ => 3000,
    }
}
/// Symbols within which 256-QAM must acquire in test 5 (skipped before counting).
const ACQUISITION_SYMBOLS: usize = 5000;
/// Symbols within which 1024-QAM must acquire in test 5 with the equalizer (skipped before
/// counting): its 16000 blind symbols, the decision mode and the first check at 20480, then the
/// lock of the carrier loop (measured, D1522).
const QAM1024_ACQUISITION_SYMBOLS: usize = 40_000;

/// Samples per symbol.
const SPS: usize = 4;
/// Symbol rate (the sample rate is `SPS` times it).
const SYMBOL_RATE: u64 = 10_000;
/// Output samples per chunk of the transmitter.
const CHUNK: usize = 1 << 14;
/// Symbols per `process` call of the receivers.
const CALL_SYMBOLS: usize = 64;
/// The clock drift of test 5 in ppm.
const DRIFT_PPM: f64 = 20.0;
/// The largest error of the offset found by the frequency acquisition (as `qam_ber_m44`).
const ACQUIRED_TOLERANCE: f64 = 1e-4;

/// `Q(x)`, the tail of the standard normal distribution.
fn q(x: f64) -> f64 {
    0.5 * libm::erfc(x / core::f64::consts::SQRT_2)
}

fn lin(db: f64) -> f64 {
    10f64.powf(db / 10.0)
}

/// The Gray-coded approximation of M44.2 test 3 for square `M`-QAM:
/// `(4/k) (1 - 1/sqrt(M)) Q(sqrt(3k/(M-1) Eb/N0))`.
fn qam_theory(order: u32, ebn0_db: f64) -> f64 {
    let m = f64::from(order);
    let k = m.log2();
    4.0 / k * (1.0 - 1.0 / m.sqrt()) * q((3.0 * k / (m - 1.0) * lin(ebn0_db)).sqrt())
}

#[test]
fn the_theory_has_its_known_values() {
    // 4-QAM at 8 dB: Q(sqrt(2 10^0.8)) = Q(3.5523) = 1.909e-4; 256-QAM at 22 dB:
    // (15/32) Q(sqrt((8/85) 10^2.2)) = (15/32) Q(3.8622) = 2.634e-5; 1024-QAM at 27 dB:
    // (31/80) Q(sqrt((10/341) 10^2.7)) = (31/80) Q(3.8337) = 2.445e-5.
    for (order, ebn0, want) in [
        (4u32, 8.0, 1.909e-4),
        (256, 22.0, 2.634e-5),
        (1024, 27.0, 2.445e-5),
    ] {
        let a = qam_theory(order, ebn0);
        assert!((a / want - 1.0).abs() < 1e-3, "{order} {ebn0}: {a}");
    }
    // 4-QAM is QPSK: the formula is Q(sqrt(2 Eb/N0)).
    let qpsk = q((2.0 * lin(8.0)).sqrt());
    assert!((qam_theory(4, 8.0) / qpsk - 1.0).abs() < 1e-12);
}

/// Random soft bits `+-1` of the seed (stream 1).
fn random_soft(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg32::new(seed, 1);
    (0..n)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

/// The symbols (`k` bits each, the first the most significant) of the soft bits `soft`.
fn symbols(soft: &[f32], k: usize) -> Vec<u32> {
    soft.chunks_exact(k)
        .map(|b| b.iter().fold(0u32, |g, &s| (g << 1) | u32::from(s > 0.0)))
        .collect()
}

/// Bits per symbol of a square order.
fn bps(order: u32) -> usize {
    order.trailing_zeros() as usize
}

/// The constellation of a square order.
fn c_of(order: u32) -> SymbolMapConstellation {
    match order {
        4 => SymbolMapConstellation::Qpsk,
        16 => SymbolMapConstellation::Qam16,
        64 => SymbolMapConstellation::Qam64,
        256 => SymbolMapConstellation::Qam256,
        1024 => SymbolMapConstellation::Qam1024,
        other => unreachable!("no square constellation of the order {other}"),
    }
}

/// The channel of a run before the noise.
#[derive(Clone, Copy, Debug)]
enum Channel {
    /// Nothing (test 3).
    Awgn,
    /// `Multipath::mild(4.0)` (test 4).
    Mild,
    /// A carrier offset in cycles per symbol and the clock drift of test 5.
    OffsetDrift(f64),
}

/// The stages of a channel, applied in order.
fn stages(channel: Channel) -> Vec<Box<dyn Impairment>> {
    let fs = (SYMBOL_RATE as usize * SPS) as f64;
    match channel {
        Channel::Awgn => Vec::new(),
        Channel::Mild => vec![Box::new(Multipath::mild(SPS as f64).unwrap())],
        Channel::OffsetDrift(offset) => vec![
            Box::new(FrequencyOffset::new(offset * SYMBOL_RATE as f64, fs).unwrap()),
            Box::new(ClockDrift::new(DRIFT_PPM).unwrap()),
        ],
    }
}

/// Runs `x` through `stages` in order.
fn apply(stages: &mut [Box<dyn Impairment>], x: &[Complex64]) -> Vec<Complex64> {
    let mut cur = x.to_vec();
    for s in stages {
        let mut next = Vec::with_capacity(cur.len() + 64);
        s.process(&cur, &mut next);
        cur = next;
    }
    cur
}

fn mod_config(order: u32) -> QamModConfig {
    QamModConfig {
        constellation: c_of(order),
        samples_per_symbol: SPS,
        ..QamModConfig::default()
    }
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

fn wide(sig: &[C32]) -> Vec<Complex64> {
    sig.iter()
        .map(|c| Complex64::new(f64::from(c.re), f64::from(c.im)))
        .collect()
}

fn narrow(x: &[Complex64]) -> Vec<C32> {
    x.iter()
        .map(|c| C32::new(c.re as f32, c.im as f32))
        .collect()
}

/// The mean power per sample of the transmitted signal after `channel` over 2^16 samples after
/// the ramp of the filter and the lag of the channel.
fn received_power(order: u32, channel: Channel, seed: u64) -> f64 {
    let b = bps(order);
    let rate = Rate::hz(SYMBOL_RATE * b as u64).unwrap();
    let mut tx = QamModBlock::<VecStorage>::new(&mod_config(order), rate, &mut VecStorage).unwrap();
    let mut sig = vec![C32::default(); (1 << 16) + 200 * SPS];
    let bits = random_soft((sig.len() / SPS + 1) * b, seed);
    let p = tx.process(&bits, &mut sig).unwrap();
    assert_eq!(p.produced, sig.len());
    let rx = apply(&mut stages(channel), &wide(&sig));
    let body = &rx[rx.len() - (1 << 16)..];
    body.iter().map(Complex64::norm_sqr).sum::<f64>() / body.len() as f64
}

/// The result of one receiver.
#[derive(Clone, Copy, Debug)]
struct Outcome {
    taps: usize,
    ber: f64,
    bits: u64,
    errors: u64,
    /// Windows of [`SLIP_WINDOW`] symbols with more than 5 % symbol errors.
    slips: u64,
    /// The rotation (quarter turns) and the lag in symbols of the alignment.
    rotation: u32,
    lag: usize,
    /// Whether the carrier loop reported lock when the first counted symbol came out.
    locked_at_start: bool,
    /// The carrier offset the frequency acquisition of the receiver found, if any.
    acquired: Option<f64>,
    /// The restarts of the receiver's controller.
    restarts: u32,
}

/// Symbols per window of the slip count.
const SLIP_WINDOW: usize = 2000;
/// Symbols compared to find the alignment.
const ALIGN_SYMBOLS: usize = 400;
/// The largest lag in symbols between the sent and the received symbols.
const MAX_LAG: usize = 80;

/// One receiver: noise, `qam_demod` and the received symbols after the skipped ones.
struct Receiver {
    order: u32,
    taps: usize,
    awgn: Awgn,
    demod: QamDemodBlock<VecStorage>,
    /// Symbols still to skip and symbols wanted.
    skip: usize,
    want: usize,
    got: Vec<u32>,
    locked_at_start: Option<bool>,
    noisy: Vec<Complex64>,
    input: Vec<C32>,
    soft: Vec<f32>,
}

impl Receiver {
    fn new(order: u32, taps: usize, ebn0_db: f64, power: f64, seed: u64, skip: usize) -> Self {
        let b = bps(order);
        Self {
            order,
            taps,
            awgn: Awgn::from_ebn0(ebn0_db, b as u32, SPS as f64, power, seed).unwrap(),
            demod: demod(order, taps),
            skip,
            want: BER_BITS as usize / b + 1 + ALIGN_SYMBOLS,
            got: Vec::new(),
            locked_at_start: None,
            noisy: Vec::with_capacity(CHUNK + 64),
            input: Vec::with_capacity(CHUNK + 64),
            soft: vec![0.0; CALL_SYMBOLS * b],
        }
    }

    fn done(&self) -> bool {
        self.got.len() >= self.want
    }

    fn push(&mut self, soft: &[f32]) {
        let mut syms = symbols(soft, bps(self.order));
        if self.skip > 0 {
            let k = self.skip.min(syms.len());
            self.skip -= k;
            syms.drain(..k);
        }
        if syms.is_empty() {
            return;
        }
        if self.locked_at_start.is_none() {
            self.locked_at_start = Some(self.demod.locked());
        }
        let room = self.want - self.got.len().min(self.want);
        self.got.extend(syms.iter().take(room));
    }

    /// Takes clean received samples (after the channel).
    fn process(&mut self, clean: &[Complex64]) {
        if self.done() {
            return;
        }
        self.noisy.clear();
        self.awgn.process(clean, &mut self.noisy);
        self.input.clear();
        self.input.extend(
            self.noisy
                .iter()
                .map(|c| C32::new(c.re as f32, c.im as f32)),
        );
        let mut at = 0;
        loop {
            let p = self
                .demod
                .process(&self.input[at..], &mut self.soft)
                .unwrap();
            at += p.consumed;
            let soft = std::mem::take(&mut self.soft);
            self.push(&soft[..p.produced]);
            self.soft = soft;
            if at == self.input.len() && p.produced < self.soft.len() {
                break;
            }
        }
    }

    /// Aligns the received symbols with the sent ones (`sent[skip + j - lag]` for the received
    /// symbol `j` turned back by the rotation, the pair with the fewest bit errors over the first
    /// [`ALIGN_SYMBOLS`]), then counts the bit errors of the next 10^6 bits.
    fn outcome(&self, sent: &[u32], skip: usize) -> Outcome {
        let b = bps(self.order);
        let half = (b / 2) as u32;
        let back = |s: u32, r: u32| rotate_symbol(s, half, (4 - r) % 4);
        let errors_at = |j: usize, r: u32, lag: usize| {
            (back(self.got[j], r) ^ sent[skip + j - lag]).count_ones() as u64
        };
        let mut best = (0u32, 0usize, u64::MAX);
        let n = ALIGN_SYMBOLS.min(self.got.len());
        for r in 0..4 {
            for lag in 0..MAX_LAG {
                let e = (0..n).map(|j| errors_at(j, r, lag)).sum();
                if e < best.2 {
                    best = (r, lag, e);
                }
            }
        }
        let (rotation, lag, _) = best;
        let counted = &self.got[n.min(self.got.len())..];
        let syms = counted.len().min(BER_BITS.div_ceil(b as u64) as usize);
        let mut errors = 0u64;
        let mut slips = 0u64;
        let mut window = 0usize;
        for j in 0..syms {
            let e = errors_at(n + j, rotation, lag);
            errors += e;
            window += usize::from(e != 0);
            if (j + 1) % SLIP_WINDOW == 0 {
                slips += u64::from(window * 20 > SLIP_WINDOW);
                window = 0;
            }
        }
        let bits = (syms * b) as u64;
        Outcome {
            taps: self.taps,
            ber: errors as f64 / bits.max(1) as f64,
            bits,
            errors,
            slips,
            rotation,
            lag,
            locked_at_start: self.locked_at_start.unwrap_or(false),
            acquired: self.demod.acquired_frequency(),
            restarts: self.demod.restarts(),
        }
    }
}

/// One receiver per entry of `runs` (`equalizer_taps`, Eb/N0) on the same transmitted signal and
/// channel, each with its own noise of `seed`; skips `skip` symbols and counts up to 10^6 bits
/// per receiver.
fn ber_run(
    order: u32,
    channel: Channel,
    seed: u64,
    runs: &[(usize, f64)],
    skip: usize,
) -> Vec<Outcome> {
    let b = bps(order);
    let power = received_power(order, channel, seed);
    // Two chunks of the transmitter more than needed (it sends whole chunks).
    let nbits =
        (skip + ALIGN_SYMBOLS + MAX_LAG + 400) * b + BER_BITS as usize + 2 * (CHUNK / SPS) * b;
    let bits = random_soft(nbits, seed);
    let sent = symbols(&bits, b);
    let rate = Rate::hz(SYMBOL_RATE * b as u64).unwrap();
    let mut tx = QamModBlock::<VecStorage>::new(&mod_config(order), rate, &mut VecStorage).unwrap();
    let mut chain = stages(channel);
    let mut rxs: Vec<Receiver> = runs
        .iter()
        .map(|&(taps, ebn0)| Receiver::new(order, taps, ebn0, power, seed, skip))
        .collect();
    let mut sig = vec![C32::default(); CHUNK];
    let per_chunk = CHUNK / SPS * b;
    let mut at = 0;
    while at + per_chunk <= bits.len() && !rxs.iter().all(Receiver::done) {
        let p = tx.process(&bits[at..at + per_chunk], &mut sig).unwrap();
        assert_eq!((p.consumed, p.produced), (per_chunk, CHUNK));
        at += per_chunk;
        let clean = apply(&mut chain, &wide(&sig));
        std::thread::scope(|s| {
            for rx in &mut rxs {
                let clean = &clean;
                s.spawn(move || rx.process(clean));
            }
        });
    }
    rxs.iter()
        .zip(runs)
        .map(|(rx, &(_, ebn0))| {
            let o = rx.outcome(&sent, skip);
            eprintln!(
                "{order}-QAM {channel:?} Eb/N0 {ebn0} dB (received power {power:.5}), \
                 equalizer_taps {}: {} errors in {} bits, BER {:.4e}, rotation {}, lag {}, slips \
                 {}, locked at the first counted symbol {}, acquired offset {:?}, restarts {}; \
                 theory {:.4e}",
                o.taps,
                o.errors,
                o.bits,
                o.ber,
                o.rotation,
                o.lag,
                o.slips,
                o.locked_at_start,
                o.acquired,
                o.restarts,
                qam_theory(order, ebn0)
            );
            o
        })
        .collect()
}

/// Asserts "within `window` dB": `BER <= theory(Eb/N0 - window)` over at least 10^6 counted bits,
/// without slips.
fn assert_within(name: &str, order: u32, o: &Outcome, ebn0_db: f64, window: f64) {
    let edge = qam_theory(order, ebn0_db - window);
    eprintln!(
        "{name}: BER {:.4e} over {} bits; theory {:.4e} at {ebn0_db} dB, edge {edge:.4e} at {} \
         dB ({:.2} errors expected at the edge)",
        o.ber,
        o.bits,
        qam_theory(order, ebn0_db),
        ebn0_db - window,
        edge * o.bits as f64
    );
    assert!(o.bits >= BER_BITS, "{name}: only {} bits counted", o.bits);
    assert_eq!(o.slips, 0, "{name}: slips");
    assert!(o.ber <= edge, "{name}: BER {:.4e} above {edge:.4e}", o.ber);
}

/// Asserts that the frequency acquisition found `offset` within [`ACQUIRED_TOLERANCE`].
fn assert_acquired(name: &str, o: &Outcome, offset: f64) {
    let a = o.acquired.unwrap_or(f64::NAN);
    assert!(
        (a - offset).abs() <= ACQUIRED_TOLERANCE,
        "{name}: acquired {a} for {offset}: {o:?}"
    );
}

/// Test 3 for one order: the receivers at `points` (Eb/N0 in dB) within `window` dB, locked at
/// the first counted symbol.
fn awgn_within(order: u32, seed: u64, points: [f64; 2], window: f64) {
    let r = ber_run(
        order,
        Channel::Awgn,
        seed,
        &points.map(|e| (11, e)),
        training_symbols(order),
    );
    for (o, e) in r.iter().zip(points) {
        let name = format!("{order}-QAM {e} dB");
        assert!(o.locked_at_start, "{name}: not locked {o:?}");
        assert_within(&name, order, o, e, window);
    }
}

/// Test 3: 4-QAM (`qpsk`) over AWGN at Eb/N0 8 and 10 dB within 1 dB of the formula.
#[test]
fn qpsk_awgn_within_1_db() {
    awgn_within(4, 825, [8.0, 10.0], 1.0);
}

/// Test 3: 256-QAM over AWGN at Eb/N0 22 and 24 dB within 1.5 dB of the formula.
#[test]
fn qam256_awgn_within_1_5_db() {
    awgn_within(256, 826, [22.0, 24.0], 1.5);
}

/// Test 3: 1024-QAM over AWGN at Eb/N0 27 and 29 dB within 2 dB of the formula.
#[test]
fn qam1024_awgn_within_2_db() {
    awgn_within(1024, 827, [27.0, 29.0], 2.0);
}

/// The Eb/N0 of the 256-QAM tests 4 and 5 (the upper point of test 3).
const QAM256_IMPAIRED_EBN0: f64 = 24.0;
/// The Eb/N0 of the 1024-QAM test 5 (the upper point of test 3).
const QAM1024_IMPAIRED_EBN0: f64 = 29.0;

/// Test 4: 256-QAM through `mild` at Eb/N0 24 dB: with `equalizer_taps` 15 within 3 dB of the
/// AWGN formula; without the equalizer the BER is above 1e-2.
#[test]
fn qam256_mild_multipath() {
    let r = ber_run(
        256,
        Channel::Mild,
        828,
        &[(15, QAM256_IMPAIRED_EBN0), (0, QAM256_IMPAIRED_EBN0)],
        training_symbols(256),
    );
    let (eq, none) = (r[0], r[1]);
    assert_within("256-QAM mild 15 taps", 256, &eq, QAM256_IMPAIRED_EBN0, 3.0);
    assert!(none.bits >= BER_BITS, "too few bits: {none:?}");
    assert!(none.ber > 1e-2, "{none:?}");
}

/// Test 5: 256-QAM with 0.001 cycles per symbol and 20 ppm at Eb/N0 24 dB acquires within 5000
/// symbols without the equalizer: lock at symbol 5000, the BER from there within 1.5 dB without
/// slips, the estimate within 1e-4.
#[test]
fn qam256_offset_and_drift_acquire_within_5000_symbols() {
    let offset = 0.001;
    let r = ber_run(
        256,
        Channel::OffsetDrift(offset),
        829,
        &[(0, QAM256_IMPAIRED_EBN0)],
        ACQUISITION_SYMBOLS,
    );
    let name = "256-QAM offset and drift";
    assert!(r[0].locked_at_start, "{name}: {:?}", r[0]);
    assert_within(name, 256, &r[0], QAM256_IMPAIRED_EBN0, 1.5);
    assert_acquired(name, &r[0], offset);
}

/// Test 5: 1024-QAM with 0.0005 cycles per symbol and 20 ppm at Eb/N0 29 dB acquires within
/// [`QAM1024_ACQUISITION_SYMBOLS`] with the equalizer: lock at that symbol, the BER from there
/// within 2 dB without slips, the estimate within 1e-4.
#[test]
fn qam1024_offset_and_drift_acquire() {
    let offset = 0.0005;
    let r = ber_run(
        1024,
        Channel::OffsetDrift(offset),
        830,
        &[(11, QAM1024_IMPAIRED_EBN0)],
        QAM1024_ACQUISITION_SYMBOLS,
    );
    let name = "1024-QAM offset and drift";
    assert!(r[0].locked_at_start, "{name}: {:?}", r[0]);
    assert_within(name, 1024, &r[0], QAM1024_IMPAIRED_EBN0, 2.0);
    assert_acquired(name, &r[0], offset);
}

/// Runs `input` through `block` in calls of 4096 samples; the soft bits.
fn demod_all(block: &mut QamDemodBlock<VecStorage>, input: &[C32]) -> Vec<f32> {
    let mut soft = Vec::new();
    let mut out = vec![0.0f32; 16_384];
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

/// The bit errors of the received symbols `rx[from..]` against `sent[j - lag]` at the alignment
/// (rotation and lag within `lead +- 3000`) with the fewest bit errors over their first
/// [`ALIGN_SYMBOLS`]: the BER over the rest and the windows of [`SLIP_WINDOW`] symbols whose
/// fewest bit errors are at another rotation or at another lag within 3 symbols.
fn errors(order: u32, sent: &[u32], rx: &[u32], from: usize, lead: usize) -> (f64, u64) {
    let k = bps(order);
    let half = (k / 2) as u32;
    let back = |s: u32, r: u32| rotate_symbol(s, half, (4 - r) % 4);
    let err = |j: usize, r: u32, o: i64| {
        usize::try_from(j as i64 - o)
            .ok()
            .and_then(|t| sent.get(t))
            .map_or(k as u32, |&t| (back(rx[j], r) ^ t).count_ones())
    };
    let lead = lead as i64;
    let mut best = (0u32, 0i64, u32::MAX);
    for r in 0..4 {
        for o in lead - 3000..lead + 3000 {
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
        bits += k as u64;
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
            .flat_map(|rr| (o - 3..=o + 3).map(move |oo| (rr, oo)))
            .filter(|&p| p != (r, o))
            .any(|(rr, oo)| count(rr, oo) < here);
        slips += u64::from(other);
    }
    (wrong as f64 / bits as f64, slips)
}

/// Test 5: the noise-first start of 256-QAM (seed 831): a carrier at 0.06 cycles per symbol for
/// 1000 symbols at the signal's amplitude (the first window takes its estimate), Gaussian noise
/// of the signal's power for 20480 symbols (the loops on noise), then the signal at 0.001 cycles
/// per symbol and Eb/N0 24 dB for 60000 symbols: the receiver restarts at least once, ends with
/// the estimate within 1e-4 of the offset, and its last 20000 symbols keep one alignment within
/// the BER of test 3.
#[test]
fn qam256_recovers_from_the_noise_first_start() {
    let (order, seed, offset, ebn0) = (256u32, 831u64, 0.001, QAM256_IMPAIRED_EBN0);
    let k = bps(order);
    let n = 60_000;
    let bits = random_soft((n + 64) * k, seed);
    let sent = symbols(&bits, k);
    let rate = Rate::hz(SYMBOL_RATE * k as u64).unwrap();
    let mut tx = QamModBlock::<VecStorage>::new(&mod_config(order), rate, &mut VecStorage).unwrap();
    let mut sig = vec![C32::default(); n * SPS];
    assert_eq!(tx.process(&bits, &mut sig).unwrap().produced, sig.len());
    let turned = apply(&mut stages(Channel::OffsetDrift(offset)), &wide(&sig));
    let body = &turned[100 * SPS..];
    let power = body.iter().map(Complex64::norm_sqr).sum::<f64>() / body.len() as f64;
    let amp = power.sqrt();
    let mut lead_in: Vec<Complex64> = (0..SPS * 1000)
        .map(|j| {
            let ph = core::f64::consts::TAU * 0.06 * j as f64 / SPS as f64;
            Complex64::new(amp * libm::cos(ph), amp * libm::sin(ph))
        })
        .collect();
    let sigma = (power / 2.0).sqrt();
    let mut rng = Pcg32::new(seed, 2);
    lead_in.extend(
        (0..SPS * 20_480).map(|_| Complex64::new(sigma * rng.normal(), sigma * rng.normal())),
    );
    let lead = lead_in.len() / SPS;
    let mut awgn = Awgn::from_ebn0(ebn0, k as u32, SPS as f64, power, seed + 100).unwrap();
    let mut noisy = Vec::with_capacity(turned.len());
    awgn.process(&turned, &mut noisy);
    let mut input = narrow(&lead_in);
    input.extend(narrow(&noisy));
    let mut block = demod(order, 11);
    let soft = demod_all(&mut block, &input);
    let rx = symbols(&soft, k);
    let (ber, slips) = errors(order, &sent, &rx, rx.len() - 20_400, lead);
    let (restarts, acquired) = (block.restarts(), block.acquired_frequency());
    let edge = qam_theory(order, ebn0 - 1.5);
    eprintln!(
        "256-QAM noise-first start: restarts {restarts}, acquired {acquired:?}, BER of the last \
         20000 symbols {ber:.4e} (edge {edge:.4e}), slips {slips}"
    );
    assert!(restarts >= 1);
    let f = acquired.unwrap();
    assert!((f - offset).abs() < ACQUIRED_TOLERANCE, "acquired {f}");
    assert_eq!(slips, 0);
    assert!(ber <= edge, "BER {ber:.4e} above {edge:.4e}");
}
