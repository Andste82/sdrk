//! M44 tests 3 to 6: `qam_mod` -> channel -> noise -> `qam_demod` (RRC matched filter,
//! `symbol_sync`, `equalize`, `costas` with the QAM constellation, the soft bits of `slice`).
//!
//! - **Test 3, 16-QAM BER (AWGN)** (seed 602): within 1 dB of the Gray-coded approximation
//!   `P_b = (4/k) (1 - 1/sqrt(M)) Q(sqrt(3k/(M-1) Eb/N0))` at Eb/N0 12 and 14 dB over 10^6 bits.
//! - **Test 4, 64-QAM BER (AWGN)** (seed 603): the same formula, within 1.5 dB at Eb/N0 18 and
//!   20 dB.
//! - **Test 5, multipath** (seed 604): 16-QAM through `Multipath::mild()` with `equalizer_taps`
//!   15 within 2.5 dB of the AWGN result (the formula of test 3, which test 3 confirms within
//!   1 dB) at Eb/N0 14 dB; without the equalizer (`equalizer_taps` 0) the BER is above 1e-2.
//! - **Test 6, offset and drift** (seed 605): 16-QAM with a carrier offset of 0.001 cycles per
//!   symbol and a clock drift of 20 ppm at Eb/N0 14 dB: the receiver acquires within 5000
//!   symbols (the carrier loop reports lock at symbol 5000, and from there the BER over 10^6
//!   bits is within 1 dB of the formula of test 3 without slips).
//!
//! "Within x dB" (as D818 of M43): the measured BER is at most the formula at `Eb/N0 - x dB`
//! over the counted bits, with no slip of the meter; there is no lower edge (the formula at
//! `Eb/N0 + x dB` expects too few errors in 10^6 bits to bound the BER from below).
//!
//! The chain (D997): random bits (`Pcg32` of the test's seed, stream 1) as soft bits `+-1` ->
//! `QamModBlock` (4 samples per symbol, roll-off 0.35, span 10) -> the channel of the test
//! (`FrequencyOffset` and `ClockDrift`, or `Multipath::mild(4.0)`, or nothing) ->
//! `Awgn::from_ebn0` (the test's seed) with the measured mean power of the signal after the
//! channel -> `QamDemodBlock` with its defaults (`equalizer_taps` 11, `carrier_bw` 0.005) and the
//! `equalizer_taps` of the test. The symbols of the first 3000 symbols (tests 3 to 5: the 1000
//! CMA symbols and 2000 in decision mode) or 5000 symbols (test 6) are skipped; the next 400
//! received symbols fix the alignment with the sent symbols (the lag of 0 to 79 symbols and the
//! rotation, the received symbol turned back by 0 to 3 quarter turns with the rotation map of
//! M44 §1, with the fewest bit errors), and the bit errors are counted over the next 10^6 bits
//! at that alignment. A window of 2000 symbols with more than 5 % symbol errors counts as a
//! slip (a lost symbol or a new rotation). Random bits instead of the PRBS-15 of the PSK tests:
//! six or four consecutive bits of PRBS-15 are linearly dependent on bits a few symbols earlier,
//! so the symbols are not independent, and the blind equalizer (CMA) converges to a wrong
//! solution on them (64-QAM at 18 dB: `mse` 0.016 instead of 0.003). The 90 degree ambiguity is
//! resolved by the alignment here and by `frame_decode` `four` in a flow.
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
use sdrk_blocks::{Block, QamDemodBlock, QamDemodConfig, QamModBlock, QamModConfig};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::rotate_symbol;
use sdrk_testkit::channel::{Awgn, ClockDrift, FrequencyOffset, Impairment, Multipath};
use sdrk_testkit::{Pcg32, VecStorage};

/// Bits counted per receiver.
const BER_BITS: u64 = 1_000_000;
/// Symbols skipped before counting in tests 3 to 5 (the training of the equalizer).
const TRAINING_SYMBOLS: usize = 3000;
/// Symbols within which test 6 must acquire (skipped before counting).
const ACQUISITION_SYMBOLS: usize = 5000;
/// Samples per symbol.
const SPS: usize = 4;
/// Symbol rate (the sample rate is `SPS` times it).
const SYMBOL_RATE: u64 = 10_000;
/// Output samples per chunk of the transmitter.
const CHUNK: usize = 1 << 14;
/// Symbols per `process` call of the receivers.
const CALL_SYMBOLS: usize = 64;
/// The carrier offset of test 6 in cycles per symbol.
const OFFSET: f64 = 0.001;
/// The clock drift of test 6 in ppm.
const DRIFT_PPM: f64 = 20.0;

/// `Q(x)`, the tail of the standard normal distribution.
fn q(x: f64) -> f64 {
    0.5 * libm::erfc(x / core::f64::consts::SQRT_2)
}

fn lin(db: f64) -> f64 {
    10f64.powf(db / 10.0)
}

/// The Gray-coded approximation of M44 test 3 for square `M`-QAM:
/// `(4/k) (1 - 1/sqrt(M)) Q(sqrt(3k/(M-1) Eb/N0))`.
fn qam_theory(order: u32, ebn0_db: f64) -> f64 {
    let m = f64::from(order);
    let k = m.log2();
    4.0 / k * (1.0 - 1.0 / m.sqrt()) * q((3.0 * k / (m - 1.0) * lin(ebn0_db)).sqrt())
}

#[test]
fn the_theory_has_its_known_values() {
    // 16-QAM at 12 dB: 0.75 Q(sqrt(0.8 10^1.2)) = 0.75 Q(3.5610) = 1.385e-4; 64-QAM at 18 dB:
    // (7/12) Q(sqrt((2/7) 10^1.8)) = (7/12) Q(4.2459) = 6.36e-6.
    let a = qam_theory(16, 12.0);
    assert!((a / 1.385e-4 - 1.0).abs() < 1e-2, "{a}");
    let b = qam_theory(64, 18.0);
    assert!((b / 6.36e-6 - 1.0).abs() < 1e-2, "{b}");
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

fn bps(order: u32) -> usize {
    if order == 64 { 6 } else { 4 }
}

/// The channel of a run before the noise.
#[derive(Clone, Copy, Debug)]
enum Channel {
    /// Nothing (tests 3 and 4).
    Awgn,
    /// `Multipath::mild(4.0)` (test 5).
    Mild,
    /// The carrier offset and the clock drift of test 6.
    OffsetDrift,
}

/// The stages of a channel, applied in order.
fn stages(channel: Channel) -> Vec<Box<dyn Impairment>> {
    let fs = (SYMBOL_RATE as usize * SPS) as f64;
    match channel {
        Channel::Awgn => Vec::new(),
        Channel::Mild => vec![Box::new(Multipath::mild(SPS as f64).unwrap())],
        Channel::OffsetDrift => vec![
            Box::new(FrequencyOffset::new(OFFSET * SYMBOL_RATE as f64, fs).unwrap()),
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
        order,
        samples_per_symbol: SPS,
        ..QamModConfig::default()
    }
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

fn wide(sig: &[C32]) -> Vec<Complex64> {
    sig.iter()
        .map(|c| Complex64::new(f64::from(c.re), f64::from(c.im)))
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
    /// Windows of [`SLIP_WINDOW`] symbols with more than 5 % symbol errors after the first
    /// one: a lost or extra symbol or a change of the rotation.
    slips: u64,
    /// The rotation (quarter turns) and the lag in symbols of the alignment.
    rotation: u32,
    lag: usize,
    /// Whether the carrier loop reported lock when the first counted symbol came out.
    locked_at_start: bool,
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
    let nbits = (skip + ALIGN_SYMBOLS + MAX_LAG + 400) * b + BER_BITS as usize + 20_000;
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
                 {}, locked at the first counted symbol {}; theory {:.4e}",
                o.taps,
                o.errors,
                o.bits,
                o.ber,
                o.rotation,
                o.lag,
                o.slips,
                o.locked_at_start,
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

/// Test 3: 16-QAM over AWGN at Eb/N0 12 and 14 dB within 1 dB of the formula.
#[test]
fn qam16_awgn_within_1_db() {
    let points = [12.0, 14.0];
    let r = ber_run(
        16,
        Channel::Awgn,
        602,
        &points.map(|e| (11, e)),
        TRAINING_SYMBOLS,
    );
    for (o, e) in r.iter().zip(points) {
        assert_within(&format!("16-QAM {e} dB"), 16, o, e, 1.0);
    }
}

/// Test 4: 64-QAM over AWGN at Eb/N0 18 and 20 dB within 1.5 dB of the formula.
#[test]
fn qam64_awgn_within_1_5_db() {
    let points = [18.0, 20.0];
    let r = ber_run(
        64,
        Channel::Awgn,
        603,
        &points.map(|e| (11, e)),
        TRAINING_SYMBOLS,
    );
    for (o, e) in r.iter().zip(points) {
        assert_within(&format!("64-QAM {e} dB"), 64, o, e, 1.5);
    }
}

/// The Eb/N0 of tests 5 and 6 (the upper point of test 3).
const IMPAIRED_EBN0: f64 = 14.0;

/// Test 5: 16-QAM through `mild` at Eb/N0 14 dB: with `equalizer_taps` 15 within 2.5 dB of the
/// AWGN formula; without the equalizer the BER is above 1e-2 (over 10^6 bits at the best
/// alignment, so that a receiver that produces nothing fails instead of passing).
#[test]
fn qam16_mild_multipath() {
    let r = ber_run(
        16,
        Channel::Mild,
        604,
        &[(15, IMPAIRED_EBN0), (0, IMPAIRED_EBN0)],
        TRAINING_SYMBOLS,
    );
    let (eq, none) = (r[0], r[1]);
    assert_within("16-QAM mild 15 taps", 16, &eq, IMPAIRED_EBN0, 2.5);
    assert!(none.bits >= BER_BITS, "too few bits: {none:?}");
    assert!(none.ber > 1e-2, "{none:?}");
}

/// Test 6: 16-QAM with 0.001 cycles per symbol of carrier offset and 20 ppm of clock drift at
/// Eb/N0 14 dB acquires within 5000 symbols: the carrier loop reports lock at symbol 5000, and
/// the BER from there is within 1 dB of the formula without slips.
#[test]
fn qam16_offset_and_drift_acquire_within_5000_symbols() {
    let r = ber_run(
        16,
        Channel::OffsetDrift,
        605,
        &[(11, IMPAIRED_EBN0)],
        ACQUISITION_SYMBOLS,
    );
    assert!(r[0].locked_at_start, "{:?}", r[0]);
    assert_within("16-QAM offset and drift", 16, &r[0], IMPAIRED_EBN0, 1.0);
}
