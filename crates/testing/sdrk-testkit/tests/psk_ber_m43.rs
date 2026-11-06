//! M43 tests 2 to 4: `psk_mod` -> `channel::Multipath` -> noise -> `psk_demod` with the
//! options `equalizer_taps` and `equalizer_feedback_taps` (the equalizer between `symbol_sync`
//! and `costas`, D825).
//!
//! - **Test 2, QPSK through `severe`** (seed 592): without the equalizer the BER at Eb/N0
//!   14 dB is above 1e-2; with `equalizer_taps` 15 and `equalizer_feedback_taps` 5 (a decision
//!   feedback equalizer, D829, D834) it is within 2 dB of the AWGN theory `Q(sqrt(2 Eb/N0))`
//!   after 3000 symbols of training. The linear equalizer with 15 taps is measured on the same
//!   signal and printed (the comparison of D834).
//! - **Test 3, 8PSK through `mild`** (seed 593): `equalizer_taps` 11, the BER at Eb/N0 16 dB
//!   within 2 dB of the AWGN result of M40.1, the union bound
//!   `(2/3) Q(sqrt(6 Eb/N0) sin(pi/8))` that M40.1 test 2 confirmed within 1 dB.
//! - **Test 4, convergence** (seed 594): QPSK through `mild` at Eb/N0 20 dB with
//!   `equalizer_taps` 11: the `mse` of the equalizer (the decision error of the
//!   phase-corrected output, D835) falls below -20 dB within the first 2000 symbols, and the
//!   phase that CMA leaves on the symbols is arbitrary but constant.
//!
//! "Within 2 dB" (D818): the measured BER is at most the theory at `Eb/N0 - 2 dB` over the
//! counted bits; as in M40 (D654) there is no lower edge, because the theory at `Eb/N0 + 2 dB`
//! expects far fewer than 10 errors in the counted bits. The equivalent Eb/N0 of the
//! equalized symbols, `10 log10(1 / (bits per symbol mean mse))` from the `mse` of the
//! equalizer averaged over the counted symbols, is printed with it as a diagnostic.
//!
//! The chain (D818): PRBS-15 bits (register all ones) as soft bits `+-1` -> `PskModBlock`
//! (4 samples per symbol, roll-off 0.35, span 10) -> `Multipath::mild(4.0)` or
//! `Multipath::severe(4.0)` -> `Awgn::from_ebn0` with the measured mean power of the
//! *received* signal (after the multipath channel) -> `PskDemodBlock` with its defaults and
//! `equalizer_taps`. No carrier offset and no clock drift (§Tests names the chain without
//! them). The soft bits of the first 3000 symbols are skipped (the 1000 CMA symbols of the
//! default and 2000 in decision mode, D828); then one `BerMeter` per phase hypothesis counts
//! until one has 10^6 bits after its synchronization (the count of M40 and M40.1); the meter
//! with the most bits is the result.
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
use sdrk_blocks::{
    Block, EqMode, PSK8_GRAY, PskDemodBlock, PskDemodConfig, PskModBlock, PskModConfig,
};
use sdrk_core::C32;
use sdrk_proto::prbs::{Poly, Prbs};
use sdrk_testkit::VecStorage;
use sdrk_testkit::ber::BerMeter;
use sdrk_testkit::channel::{Awgn, Impairment, Multipath};

/// Bits counted per receiver.
const BER_BITS: u64 = 1_000_000;
/// Symbols skipped before counting (training).
const TRAINING_SYMBOLS: usize = 3000;
/// The feedback taps of test 2 (D834).
const SEVERE_FEEDBACK: usize = 5;
/// Samples per symbol.
const SPS: usize = 4;
/// Symbol rate (the sample rate is `SPS` times it).
const SYMBOL_RATE: u64 = 10_000;
/// Output samples per chunk of the transmitter.
const CHUNK: usize = 1 << 14;
/// Symbols per `process` call of the receivers, so that `mse` is sampled often.
const CALL_SYMBOLS: usize = 64;
/// Eb/N0 of tests 2, 3 and 4.
const SEVERE_EBN0: f64 = 14.0;
const MILD_EBN0: f64 = 16.0;
const CONVERGENCE_EBN0: f64 = 20.0;
/// The window of "within 2 dB".
const WINDOW_DB: f64 = 2.0;

/// `Q(x)`, the tail of the standard normal distribution.
fn q(x: f64) -> f64 {
    0.5 * libm::erfc(x / core::f64::consts::SQRT_2)
}

fn lin(db: f64) -> f64 {
    10f64.powf(db / 10.0)
}

/// Coherent Gray QPSK: `Q(sqrt(2 Eb/N0))`.
fn qpsk_theory(ebn0_db: f64) -> f64 {
    q((2.0 * lin(ebn0_db)).sqrt())
}

/// Gray 8PSK, union-bound approximation: `(2/3) Q(sqrt(6 Eb/N0) sin(pi/8))` (M40.1).
fn psk8_theory(ebn0_db: f64) -> f64 {
    2.0 / 3.0 * q((6.0 * lin(ebn0_db)).sqrt() * (core::f64::consts::PI / 8.0).sin())
}

#[test]
fn the_theories_have_their_known_values() {
    // Q(sqrt(2 10^1.2)) = Q(5.6308) = 8.98e-9; (2/3) Q(sqrt(6 10^1.4) sin(pi/8)) = 8.7e-7.
    assert!(
        (qpsk_theory(12.0) / 8.98e-9 - 1.0).abs() < 5e-3,
        "{}",
        qpsk_theory(12.0)
    );
    assert!(
        (psk8_theory(14.0) / 8.7e-7 - 1.0).abs() < 1e-2,
        "{}",
        psk8_theory(14.0)
    );
}

/// Soft bits `+-1` of PRBS-15 (register all ones).
fn prbs_soft(n: usize) -> Vec<f32> {
    let mut p = Prbs::new(Poly::Prbs15, 0x7fff).unwrap();
    (0..n)
        .map(|_| if p.next_bit() == 1 { 1.0 } else { -1.0 })
        .collect()
}

/// The Gray code `g` of an 8PSK symbol received turned by `r` eighth turns, mapped back to
/// the transmitted code (as in `psk_ber_m40_1.rs`).
fn unturn8(g: usize, r: usize) -> usize {
    let k = usize::from(PSK8_GRAY[g]);
    let t = (k + 8 - r) % 8;
    PSK8_GRAY.iter().position(|&x| usize::from(x) == t).unwrap()
}

/// The bit pair of a QPSK symbol received turned by `r` quarter turns, mapped back (as in
/// `psk_ber_m40.rs`).
fn unturn(mut b0: bool, mut b1: bool, r: usize) -> (bool, bool) {
    for _ in 0..r {
        (b0, b1) = (b1, !b0);
    }
    (b0, b1)
}

/// The channel preset of a run.
#[derive(Clone, Copy, Debug)]
enum Preset {
    Mild,
    Severe,
}

impl Preset {
    fn channel(self) -> Multipath {
        match self {
            Self::Mild => Multipath::mild(SPS as f64).unwrap(),
            Self::Severe => Multipath::severe(SPS as f64).unwrap(),
        }
    }
}

fn mod_config(order: u32) -> PskModConfig {
    PskModConfig {
        order,
        samples_per_symbol: SPS,
        ..PskModConfig::default()
    }
}

fn demod(order: u32, taps: usize, feedback: usize) -> PskDemodBlock<VecStorage> {
    let cfg = PskDemodConfig {
        order,
        symbol_rate_hz: SYMBOL_RATE as f64,
        equalizer_taps: taps,
        equalizer_feedback_taps: feedback,
        ..PskDemodConfig::default()
    };
    PskDemodBlock::new(
        &cfg,
        Rate::hz(SYMBOL_RATE * SPS as u64).unwrap(),
        &mut VecStorage,
    )
    .unwrap()
}

fn bps(order: u32) -> usize {
    if order == 8 { 3 } else { 2 }
}

/// The mean power per sample of the transmitted signal after `preset` over 2^16 samples
/// after the ramp of the filter and the lag of the channel.
fn received_power(order: u32, preset: Preset) -> f64 {
    let b = bps(order);
    let rate = Rate::hz(SYMBOL_RATE * b as u64).unwrap();
    let mut tx = PskModBlock::<VecStorage>::new(&mod_config(order), rate, &mut VecStorage).unwrap();
    let mut sig = vec![C32::default(); (1 << 16) + 200 * SPS];
    let bits = prbs_soft((sig.len() / SPS + 1) * b);
    let p = tx.process(&bits, &mut sig).unwrap();
    assert_eq!(p.produced, sig.len());
    let wide: Vec<Complex64> = sig
        .iter()
        .map(|c| Complex64::new(f64::from(c.re), f64::from(c.im)))
        .collect();
    let mut rx = Vec::new();
    preset.channel().process(&wide, &mut rx);
    let body = &rx[rx.len() - (1 << 16)..];
    body.iter().map(Complex64::norm_sqr).sum::<f64>() / body.len() as f64
}

/// The result of one receiver.
#[derive(Clone, Copy, Debug)]
struct Outcome {
    taps: usize,
    feedback: usize,
    ber: f64,
    bits: u64,
    errors: u64,
    slips: u64,
    hypothesis: usize,
    /// The bits pushed into the meters after the training (per meter).
    pushed: u64,
    /// The mean `mse` of the equalizer over the counted symbols (NaN without equalizer).
    mse: f64,
}

/// One receiver: noise, `psk_demod` and a meter per phase hypothesis.
struct Receiver {
    order: u32,
    taps: usize,
    feedback: usize,
    awgn: Awgn,
    demod: PskDemodBlock<VecStorage>,
    meters: Vec<BerMeter>,
    budget: u64,
    skip: usize,
    /// The bits pushed into each meter after the training.
    pushed: u64,
    mse_sum: f64,
    mse_n: u64,
    noisy: Vec<Complex64>,
    input: Vec<C32>,
    soft: Vec<f32>,
}

impl Receiver {
    fn new(
        order: u32,
        eq: (usize, usize),
        ebn0_db: f64,
        power: f64,
        seed: u64,
        budget: u64,
    ) -> Self {
        let b = bps(order);
        Self {
            order,
            taps: eq.0,
            feedback: eq.1,
            awgn: Awgn::from_ebn0(ebn0_db, b as u32, SPS as f64, power, seed).unwrap(),
            demod: demod(order, eq.0, eq.1),
            meters: (0..order)
                .map(|_| BerMeter::new(Poly::Prbs15, 0x7fff).unwrap())
                .collect(),
            budget,
            skip: TRAINING_SYMBOLS * b,
            pushed: 0,
            mse_sum: 0.0,
            mse_n: 0,
            noisy: Vec::with_capacity(CHUNK + 64),
            input: Vec::with_capacity(CHUNK + 64),
            soft: vec![0.0; CALL_SYMBOLS * b],
        }
    }

    fn done(&self) -> bool {
        self.meters.iter().any(|m| m.bits() >= self.budget)
    }

    fn push(&mut self, soft: &[f32]) {
        let soft = if self.skip > 0 {
            let k = self.skip.min(soft.len());
            self.skip -= k;
            &soft[k..]
        } else {
            soft
        };
        if soft.is_empty() {
            return;
        }
        if let Some(eq) = self.demod.equalizer() {
            self.mse_sum += f64::from(eq.mse());
            self.mse_n += 1;
        }
        for sym in soft.chunks_exact(bps(self.order)) {
            if self.done() {
                return;
            }
            self.pushed += sym.len() as u64;
            if self.order == 8 {
                let g = sym.iter().fold(0, |g, &s| (g << 1) | usize::from(s > 0.0));
                for (r, m) in self.meters.iter_mut().enumerate() {
                    let t = unturn8(g, r);
                    m.push(u8::from(t & 4 != 0));
                    m.push(u8::from(t & 2 != 0));
                    m.push(u8::from(t & 1 != 0));
                }
            } else {
                let (b0, b1) = (sym[0] > 0.0, sym[1] > 0.0);
                for (r, m) in self.meters.iter_mut().enumerate() {
                    let (c0, c1) = unturn(b0, b1, r);
                    m.push(u8::from(c0));
                    m.push(u8::from(c1));
                }
            }
        }
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

    fn outcome(&self) -> Outcome {
        let (r, best) = self
            .meters
            .iter()
            .enumerate()
            .max_by_key(|(_, m)| m.bits())
            .unwrap();
        Outcome {
            taps: self.taps,
            feedback: self.feedback,
            ber: best.ber(),
            bits: best.bits(),
            errors: best.errors(),
            slips: self.meters.iter().map(BerMeter::slips).sum(),
            hypothesis: r,
            pushed: self.pushed,
            mse: if self.mse_n == 0 {
                f64::NAN
            } else {
                self.mse_sum / self.mse_n as f64
            },
        }
    }
}

/// Runs one receiver per entry of `taps` (forward and feedback taps; 0 forward taps = no
/// equalizer) on the same transmitted signal, channel and noise; counts up to `budget` bits per
/// receiver.
fn ber_run(
    order: u32,
    preset: Preset,
    ebn0_db: f64,
    seed: u64,
    taps: &[(usize, usize)],
    budget: u64,
) -> Vec<Outcome> {
    let b = bps(order);
    let power = received_power(order, preset);
    let nbits = (TRAINING_SYMBOLS + 400) * b + budget as usize + 20_000;
    let bits = prbs_soft(nbits);
    let rate = Rate::hz(SYMBOL_RATE * b as u64).unwrap();
    let mut tx = PskModBlock::<VecStorage>::new(&mod_config(order), rate, &mut VecStorage).unwrap();
    let mut channel = preset.channel();
    let mut rxs: Vec<Receiver> = taps
        .iter()
        .map(|&t| Receiver::new(order, t, ebn0_db, power, seed, budget))
        .collect();
    let mut sig = vec![C32::default(); CHUNK];
    let mut wide = Vec::with_capacity(CHUNK);
    let mut clean = Vec::with_capacity(CHUNK);
    let per_chunk = CHUNK / SPS * b;
    let mut at = 0;
    while at + per_chunk <= bits.len() && !rxs.iter().all(Receiver::done) {
        let p = tx.process(&bits[at..at + per_chunk], &mut sig).unwrap();
        assert_eq!((p.consumed, p.produced), (per_chunk, CHUNK));
        at += per_chunk;
        wide.clear();
        wide.extend(
            sig.iter()
                .map(|c| Complex64::new(f64::from(c.re), f64::from(c.im))),
        );
        clean.clear();
        channel.process(&wide, &mut clean);
        std::thread::scope(|s| {
            for rx in &mut rxs {
                let clean = &clean;
                s.spawn(move || rx.process(clean));
            }
        });
    }
    rxs.iter()
        .map(|rx| {
            let o = rx.outcome();
            eprintln!(
                "order {order} {preset:?} Eb/N0 {ebn0_db} dB (received power {power:.5}), \
                 equalizer_taps {} feedback {}: {} errors in {} bits, BER {:.4e}, hypothesis {}, \
                 slips {}, mean mse {:.5} ({:.2} dB), equivalent Eb/N0 {:.2} dB",
                o.taps,
                o.feedback,
                o.errors,
                o.bits,
                o.ber,
                o.hypothesis,
                o.slips,
                o.mse,
                10.0 * o.mse.log10(),
                10.0 * (1.0 / (b as f64 * o.mse)).log10(),
            );
            o
        })
        .collect()
}

/// Asserts "within 2 dB" (D818): `BER <= theory(Eb/N0 - 2 dB)` over at least `BER_BITS`
/// counted bits, without slips.
fn assert_within_2_db(name: &str, o: &Outcome, ebn0_db: f64, theory: fn(f64) -> f64) {
    let edge = theory(ebn0_db - WINDOW_DB);
    eprintln!(
        "{name}: BER {:.4e} over {} bits; theory {:.4e} at {ebn0_db} dB, edge {edge:.4e} at \
         {} dB ({:.2} errors expected at the edge)",
        o.ber,
        o.bits,
        theory(ebn0_db),
        ebn0_db - WINDOW_DB,
        edge * o.bits as f64
    );
    assert!(o.bits >= BER_BITS, "{name}: only {} bits counted", o.bits);
    assert_eq!(o.slips, 0, "{name}: slips");
    assert!(o.ber <= edge, "{name}: BER {:.4e} above {edge:.4e}", o.ber);
}

/// Test 2, first part: QPSK through `severe` at Eb/N0 14 dB without the equalizer has a BER
/// above 1e-2. The receiver must have pushed at least `BER_BITS / 2` bits after the training
/// and a meter must have synchronized and counted bits, so that a receiver that produces no
/// soft bits or never synchronizes fails instead of passing (D824). The linear equalizer with
/// `equalizer_taps` 15 is measured on the same signal and printed (the comparison with the
/// decision feedback equalizer of the test below, D834).
#[test]
fn qpsk_severe_without_equalizer_is_above_1e_2() {
    let r = ber_run(
        4,
        Preset::Severe,
        SEVERE_EBN0,
        592,
        &[(0, 0), (15, 0)],
        BER_BITS,
    );
    let (none, eq) = (r[0], r[1]);
    eprintln!(
        "qpsk severe: without equalizer BER {:.4e} ({} bits, {} slips); linear 15 taps BER \
         {:.4e} ({} slips); theory {:.4e}, 2 dB edge {:.4e}",
        none.ber,
        none.bits,
        none.slips,
        eq.ber,
        eq.slips,
        qpsk_theory(SEVERE_EBN0),
        qpsk_theory(SEVERE_EBN0 - WINDOW_DB)
    );
    assert!(none.pushed >= BER_BITS / 2, "too few bits pushed: {none:?}");
    assert!(none.bits > 0, "no meter synchronized: {none:?}");
    assert!(none.ber > 1e-2, "{none:?}");
}

/// Test 2, second part: with `equalizer_taps` 15 and `equalizer_feedback_taps` 5 the BER is
/// within 2 dB of `Q(sqrt(2 Eb/N0))` at 14 dB after 3000 training symbols (D818: at most the
/// theory at 12 dB, 9.0e-9, over 10^6 counted bits without slips, so no error at all; D834).
/// `severe` has a spectral null of -30 dB inside the signal band, where a linear equalizer
/// enhances the noise (about 6 dB loss with the optimum 15 taps, I43-1); the decision feedback
/// cancels the postcursors with the past decisions instead.
#[test]
fn qpsk_severe_with_15_taps_within_2_db() {
    let r = ber_run(
        4,
        Preset::Severe,
        SEVERE_EBN0,
        592,
        &[(15, SEVERE_FEEDBACK)],
        BER_BITS,
    );
    assert_within_2_db("qpsk severe 15 + 5 taps", &r[0], SEVERE_EBN0, qpsk_theory);
}

/// Test 3: 8PSK through `mild` with `equalizer_taps` 11 at Eb/N0 16 dB, within 2 dB of the
/// AWGN result of M40.1 (its union-bound theory, which M40.1 test 2 met within 1 dB at 12 and
/// 14 dB; M40.1 measured no point at 16 dB). With the equalizer after the carrier loop (the
/// first order of section 2) the order-8 loop slipped on the unequalized symbols (I43-3); before
/// it (D825) the decisions see equalized symbols.
#[test]
fn psk8_mild_with_11_taps_within_2_db() {
    let r = ber_run(8, Preset::Mild, MILD_EBN0, 593, &[(11, 0)], BER_BITS);
    assert_within_2_db("8psk mild 11 taps", &r[0], MILD_EBN0, psk8_theory);
}

/// Symbols of the convergence test.
const CONVERGENCE_SYMBOLS: usize = 8000;
/// The symbols within which `mse` must fall below -20 dB.
const CONVERGENCE_LIMIT: usize = 2000;
/// The CMA symbols of the default (`PskDemodConfig::equalizer`, D828).
const CMA_SYMBOLS: usize = 1000;
/// Symbols between two readings of the CMA phase.
const PHASE_STEP: usize = 50;
/// The first reading of the CMA phase (after the first CMA steps).
const PHASE_START: usize = 250;
/// The largest spread of the CMA phase readings (D819, D835).
const PHASE_SPREAD: f64 = 0.05;
/// The centre tap of 11.
const CENTRE: usize = 5;

/// The measurements of the convergence run.
#[derive(Debug)]
struct Convergence {
    /// The first symbol after which `mse` is below -20 dB (0.01), over the whole run.
    first: Option<usize>,
    /// The smallest `mse` over the first [`CONVERGENCE_LIMIT`] symbols.
    min_mse: f32,
    /// The `mse` after the run.
    final_mse: f32,
    /// The phase of the centre weight every [`PHASE_STEP`] symbols from [`PHASE_START`] to the
    /// end of the CMA mode.
    phases: Vec<f64>,
}

impl Convergence {
    fn spread(&self) -> f64 {
        let lo = self.phases.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = self
            .phases
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    }

    fn print(&self, n: usize) {
        eprintln!(
            "convergence (mild, Eb/N0 {CONVERGENCE_EBN0} dB, 11 taps, {n} symbols): mse below \
             -20 dB first after symbol {:?}; smallest mse in the first {CONVERGENCE_LIMIT} \
             symbols {:.5} ({:.2} dB); final mse {:.5} ({:.2} dB); CMA phases {:.4?} rad, \
             spread {:.4} rad",
            self.first,
            self.min_mse,
            10.0 * f64::from(self.min_mse).log10(),
            self.final_mse,
            10.0 * f64::from(self.final_mse).log10(),
            self.phases,
            self.spread()
        );
    }
}

/// Test 4's run: QPSK through `mild` at Eb/N0 20 dB (seed 594) into `psk_demod` with
/// `equalizer_taps` 11 and the defaults of `equalize` (`mu_cma` 0.05, `mu_dd` 0.02, 1000 CMA
/// symbols, the phase tracker at `carrier_bw` 0.02), one symbol per call; `mse` and the weights
/// are read after every symbol. Also checks the switch to decision mode after the 1000th symbol
/// (D805).
fn convergence_run() -> Convergence {
    let order = 4;
    let power = received_power(order, Preset::Mild);
    let bits = prbs_soft((CONVERGENCE_SYMBOLS + 100) * 2);
    let rate = Rate::hz(SYMBOL_RATE * 2).unwrap();
    let mut tx = PskModBlock::<VecStorage>::new(&mod_config(order), rate, &mut VecStorage).unwrap();
    let mut sig = vec![C32::default(); (CONVERGENCE_SYMBOLS + 100) * SPS];
    let p = tx.process(&bits, &mut sig).unwrap();
    assert_eq!(p.produced, sig.len());
    let wide: Vec<Complex64> = sig
        .iter()
        .map(|c| Complex64::new(f64::from(c.re), f64::from(c.im)))
        .collect();
    let mut clean = Vec::new();
    Preset::Mild.channel().process(&wide, &mut clean);
    let mut noisy = Vec::new();
    Awgn::from_ebn0(CONVERGENCE_EBN0, 2, SPS as f64, power, 594)
        .unwrap()
        .process(&clean, &mut noisy);
    let input: Vec<C32> = noisy
        .iter()
        .map(|c| C32::new(c.re as f32, c.im as f32))
        .collect();
    let mut rx = demod(order, 11, 0);
    let mut soft = [0.0f32; 8];
    let mut n = 0usize;
    let mut first = None;
    let mut min_mse = f32::INFINITY;
    let mut phases = Vec::new();
    let mut next_reading = PHASE_START;
    for piece in input.chunks(SPS) {
        let p = rx.process(piece, &mut soft).unwrap();
        assert_eq!(p.consumed, piece.len());
        n += p.produced / 2;
        let eq = rx.equalizer().unwrap();
        let expected = if n < CMA_SYMBOLS {
            EqMode::Cma
        } else {
            EqMode::Decision
        };
        assert_eq!(eq.mode(), expected, "symbol {n}");
        if n <= CONVERGENCE_LIMIT {
            min_mse = min_mse.min(eq.mse());
        }
        if first.is_none() && eq.mse() < 0.01 {
            first = Some(n);
        }
        // The reading after the symbol that reaches the next multiple of 50 (a call can give two
        // symbols when the timing loop slips a sample).
        if n >= next_reading && next_reading <= CMA_SYMBOLS {
            let w = eq.weights()[CENTRE];
            phases.push(libm::atan2(f64::from(w.im), f64::from(w.re)));
            next_reading += PHASE_STEP;
        }
    }
    let c = Convergence {
        first,
        min_mse,
        final_mse: rx.equalizer().unwrap().mse(),
        phases,
    };
    c.print(n);
    c
}

/// Test 4, the phase: the phase that CMA leaves on the symbols is the common phase of the
/// weights (the channel has no carrier phase, and the CMA error does not depend on it); the phase
/// of the centre weight, read every 50 symbols from symbol 250 to the end of the CMA mode
/// (1000), is arbitrary but constant: all readings within 0.05 rad (D819, D835).
#[test]
fn cma_phase_on_mild_is_constant() {
    let c = convergence_run();
    assert_eq!(c.phases.len(), 16, "{c:?}");
    assert!(
        c.spread() <= PHASE_SPREAD,
        "phase spread {} rad",
        c.spread()
    );
}

/// Test 4, the convergence: `mse` (the decision error power of the output that the phase
/// tracker corrected, D835) falls below -20 dB within 2000 symbols.
#[test]
fn mse_on_mild_below_minus_20_db_within_2000_symbols() {
    let c = convergence_run();
    let first = c.first.expect("mse never fell below -20 dB");
    assert!(
        first <= CONVERGENCE_LIMIT,
        "first below -20 dB after symbol {first}"
    );
}
