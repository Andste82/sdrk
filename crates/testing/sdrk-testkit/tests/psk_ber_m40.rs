//! M40 tests 2 and 3: the bit error rate of `psk_mod` -> channel -> `psk_demod`.
//!
//! - **Test 2, coherent** (BPSK seed 553, QPSK seed 554): within a factor 1.5 of
//!   `Q(sqrt(2 Eb/N0))` at Eb/N0 6 and 8 dB.
//! - **Test 3, differential** (DBPSK seed 555, DQPSK seed 556): within 1 dB of the theory
//!   (`0.5 exp(-Eb/N0)` for DBPSK, the exact Gray DQPSK formula with the Marcum Q function for
//!   DQPSK, D619) at Eb/N0 6 and 8 dB: the measured BER lies between the theory at Eb/N0 + 1 dB
//!   and at Eb/N0 - 1 dB.
//!
//! The chain (D618): PRBS-15 bits (register all ones) as soft bits `+-1` -> `PskModBlock`
//! (4 samples per symbol, roll-off 0.35, span 10) -> `FrequencyOffset` of 0.002 cycles per
//! symbol -> `ClockDrift` of 50 ppm -> `Awgn::from_ebn0` (the measured mean power of the
//! transmitted signal, the bits per symbol of the order, 4 samples per symbol) ->
//! `PskDemodBlock` with its defaults. The soft bits of the first 2000 symbols are skipped
//! (acquisition); then `BerMeter`s count until one has 10^6 bits after its synchronization.
//! The ambiguity of coherent detection is removed by one meter per hypothesis: both
//! polarities for BPSK, the four rotations for QPSK (a turn by 90 degrees maps the bit pair
//! `(b0, b1)` to `(!b1, b0)`); the meter that synchronizes is the result, and no meter may
//! report a slip. The transmitter, the offset and the drift are computed once per test and
//! shared by the receivers of both Eb/N0 points, which run on two threads.
//!
//! The measured values are printed (run with `--nocapture`).
#![cfg(feature = "std")]
#![allow(
    clippy::cast_lossless,
    clippy::manual_midpoint,
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
use sdrk_blocks::{Block, PskDemodBlock, PskDemodConfig, PskModBlock, PskModConfig};
use sdrk_core::C32;
use sdrk_proto::prbs::{Poly, Prbs};
use sdrk_testkit::VecStorage;
use sdrk_testkit::ber::BerMeter;
use sdrk_testkit::channel::{Awgn, ClockDrift, FrequencyOffset, Impairment};

/// Bits counted per Eb/N0 point.
const BER_BITS: u64 = 1_000_000;
/// Symbols skipped before counting (acquisition).
const ACQUISITION_SYMBOLS: usize = 2000;
/// Samples per symbol of the tests.
const SPS: usize = 4;
/// Symbol rate (the sample rate is `SPS` times it).
const SYMBOL_RATE: u64 = 10_000;
/// Carrier offset in cycles per symbol.
const OFFSET: f64 = 0.002;
/// Clock drift in ppm.
const DRIFT_PPM: f64 = 50.0;
/// Input samples per chunk of the transmitter.
const CHUNK: usize = 1 << 14;
/// The Eb/N0 points in dB.
const POINTS: [f64; 2] = [6.0, 8.0];

/// `Q(x)`, the tail of the standard normal distribution.
fn q(x: f64) -> f64 {
    0.5 * libm::erfc(x / core::f64::consts::SQRT_2)
}

fn lin(db: f64) -> f64 {
    10f64.powf(db / 10.0)
}

/// Coherent BPSK and Gray QPSK: `Q(sqrt(2 Eb/N0))`.
fn coherent_theory(ebn0_db: f64) -> f64 {
    q((2.0 * lin(ebn0_db)).sqrt())
}

/// DBPSK: `0.5 exp(-Eb/N0)`.
fn dbpsk_theory(ebn0_db: f64) -> f64 {
    0.5 * (-lin(ebn0_db)).exp()
}

/// `I0(z) exp(-z)` from the power series (`z` up to a few hundred).
fn i0_scaled(z: f64) -> f64 {
    // Sum the series in a scaled form: terms (z/2)^(2k) / (k!)^2 exp(-z).
    let mut term = (-z).exp();
    let mut sum = term;
    let h = 0.25 * z * z;
    let mut k = 1.0;
    while term > sum * 1e-17 || k < 2.0 * z + 10.0 {
        term *= h / (k * k);
        sum += term;
        k += 1.0;
        if k > 5000.0 {
            break;
        }
    }
    sum
}

/// The Marcum Q function `Q1(a, b) = int_b^inf x exp(-(x^2 + a^2) / 2) I0(a x) dx`
/// (Simpson's rule on `[b, b + 40]`).
fn marcum_q1(a: f64, b: f64) -> f64 {
    let n = 40_000;
    let h = 40.0 / n as f64;
    let f = |x: f64| x * (-(x - a) * (x - a) / 2.0).exp() * i0_scaled(a * x);
    let mut s = f(b) + f(b + 40.0);
    for i in 1..n {
        let x = b + i as f64 * h;
        s += if i % 2 == 1 { 4.0 } else { 2.0 } * f(x);
    }
    s * h / 3.0
}

/// Gray-coded DQPSK with differential detection (Proakis): `Pb = Q1(a, b) - 0.5 I0(a b)
/// exp(-(a^2 + b^2) / 2)` with `a = sqrt(2 g (1 - 1/sqrt 2))`, `b = sqrt(2 g (1 + 1/sqrt 2))`
/// and `g = Eb/N0`.
fn dqpsk_theory(ebn0_db: f64) -> f64 {
    let g = lin(ebn0_db);
    let r = core::f64::consts::FRAC_1_SQRT_2;
    let a = (2.0 * g * (1.0 - r)).sqrt();
    let b = (2.0 * g * (1.0 + r)).sqrt();
    marcum_q1(a, b) - 0.5 * i0_scaled(a * b) * (a * b - (a * a + b * b) / 2.0).exp()
}

#[test]
fn the_theories_have_their_known_values() {
    // Coherent BPSK at 6 and 8 dB: 2.388e-3 and 1.909e-4.
    assert!((coherent_theory(6.0) / 2.388_2e-3 - 1.0).abs() < 1e-3);
    assert!((coherent_theory(8.0) / 1.909_1e-4 - 1.0).abs() < 1e-3);
    assert!((dbpsk_theory(8.0) - 0.5 * (-6.309_573_444_801_933f64).exp()).abs() < 1e-15);
    // Q1(0, b) = exp(-b^2 / 2); I0 scaled at 0 is 1.
    assert!((marcum_q1(0.0, 1.5) - (-1.125f64).exp()).abs() < 1e-9);
    assert!((i0_scaled(0.0) - 1.0).abs() < 1e-15);
    // I0(10) = 2815.716628466254
    assert!((i0_scaled(10.0) * 10f64.exp() / 2_815.716_628_466_254 - 1.0).abs() < 1e-12);
    // DQPSK is about 2 dB worse than coherent QPSK at these error rates.
    for db in [6.0, 8.0] {
        let d = dqpsk_theory(db);
        assert!(
            d > coherent_theory(db) && d < coherent_theory(db - 3.0),
            "{db}: {d}"
        );
    }
}

/// Soft bits `+-1` of PRBS-15 (register all ones).
fn prbs_soft(n: usize) -> Vec<f32> {
    let mut p = Prbs::new(Poly::Prbs15, 0x7fff).unwrap();
    (0..n)
        .map(|_| if p.next_bit() == 1 { 1.0 } else { -1.0 })
        .collect()
}

/// The bit pair `(b0, b1)` of a QPSK symbol received turned by `r` quarter turns
/// counter-clockwise, mapped back: one turn gives `(!b1, b0)`.
fn unturn(mut b0: bool, mut b1: bool, r: usize) -> (bool, bool) {
    for _ in 0..r {
        (b0, b1) = (b1, !b0);
    }
    (b0, b1)
}

/// One receiver: noise, `psk_demod` and a meter per phase hypothesis.
struct Receiver {
    ebn0_db: f64,
    awgn: Awgn,
    demod: PskDemodBlock<VecStorage>,
    meters: Vec<BerMeter>,
    order: u32,
    skip: usize,
    noisy: Vec<Complex64>,
    input: Vec<C32>,
    soft: Vec<f32>,
}

impl Receiver {
    fn new(order: u32, differential: bool, ebn0_db: f64, power: f64, seed: u64) -> Self {
        let bps = if order == 4 { 2 } else { 1 };
        let cfg = PskDemodConfig {
            order,
            symbol_rate_hz: SYMBOL_RATE as f64,
            differential,
            ..PskDemodConfig::default()
        };
        let fs = Rate::hz(SYMBOL_RATE * SPS as u64).unwrap();
        let hypotheses = match (differential, order) {
            (true, _) => 1,
            (false, 2) => 2,
            _ => 4,
        };
        Self {
            ebn0_db,
            awgn: Awgn::from_ebn0(ebn0_db, bps, SPS as f64, power, seed).unwrap(),
            demod: PskDemodBlock::new(&cfg, fs, &mut VecStorage).unwrap(),
            meters: (0..hypotheses)
                .map(|_| BerMeter::new(Poly::Prbs15, 0x7fff).unwrap())
                .collect(),
            order,
            skip: ACQUISITION_SYMBOLS * bps as usize,
            noisy: Vec::with_capacity(CHUNK + 64),
            input: Vec::with_capacity(CHUNK + 64),
            soft: vec![0.0; CHUNK],
        }
    }

    fn done(&self) -> bool {
        self.meters.iter().any(|m| m.bits() >= BER_BITS)
    }

    fn push(&mut self, soft: &[f32]) {
        let soft = if self.skip > 0 {
            let k = self.skip.min(soft.len());
            self.skip -= k;
            &soft[k..]
        } else {
            soft
        };
        if self.order == 4 {
            for pair in soft.chunks_exact(2) {
                if self.done() {
                    return;
                }
                let (b0, b1) = (pair[0] > 0.0, pair[1] > 0.0);
                for (r, m) in self.meters.iter_mut().enumerate() {
                    let (c0, c1) = unturn(b0, b1, r);
                    m.push(u8::from(c0));
                    m.push(u8::from(c1));
                }
            }
        } else {
            for &s in soft {
                if self.done() {
                    return;
                }
                let b = s > 0.0;
                for (r, m) in self.meters.iter_mut().enumerate() {
                    m.push(u8::from(b != (r == 1)));
                }
            }
        }
    }

    /// Takes clean received samples (after offset and drift).
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

    /// The meter that synchronized (the most bits), and the slips of all meters.
    fn result(&self) -> (&BerMeter, u64) {
        let best = self.meters.iter().max_by_key(|m| m.bits()).unwrap();
        (best, self.meters.iter().map(BerMeter::slips).sum())
    }
}

/// `(Eb/N0, BER, bits, slips)` of every point.
fn ber_run(order: u32, differential: bool, seed: u64) -> Vec<(f64, f64, u64, u64)> {
    let start = std::time::Instant::now();
    let bps = if order == 4 { 2usize } else { 1 };
    let m = PskModConfig {
        order,
        samples_per_symbol: SPS,
        differential,
        ..PskModConfig::default()
    };
    // Bits for the acquisition, the counted bits, the meter synchronization and the delays.
    let nbits = (ACQUISITION_SYMBOLS + 200) * bps + BER_BITS as usize + 10_000;
    let bits = prbs_soft(nbits);
    let bit_rate = Rate::hz(SYMBOL_RATE * bps as u64).unwrap();

    // The mean power per sample of the transmitted signal over 2^16 samples after the ramp.
    let mut probe_tx = PskModBlock::<VecStorage>::new(&m, bit_rate, &mut VecStorage).unwrap();
    let mut probe = vec![C32::default(); (1 << 16) + 20 * SPS];
    let p = probe_tx.process(&bits, &mut probe).unwrap();
    assert_eq!(p.produced, probe.len());
    let power = probe[20 * SPS..]
        .iter()
        .map(|c| f64::from(c.re).powi(2) + f64::from(c.im).powi(2))
        .sum::<f64>()
        / (1 << 16) as f64;
    eprintln!("order {order} differential {differential}: mean signal power {power:.5}");

    let mut tx = PskModBlock::<VecStorage>::new(&m, bit_rate, &mut VecStorage).unwrap();
    let fs = (SYMBOL_RATE as usize * SPS) as f64;
    let mut offset = FrequencyOffset::new(OFFSET * SYMBOL_RATE as f64, fs).unwrap();
    let mut drift = ClockDrift::new(DRIFT_PPM).unwrap();
    let mut rxs: Vec<Receiver> = POINTS
        .iter()
        .map(|&e| Receiver::new(order, differential, e, power, seed))
        .collect();
    let mut sig = vec![C32::default(); CHUNK];
    let mut wide = Vec::with_capacity(CHUNK);
    let mut shifted = Vec::with_capacity(CHUNK);
    let mut clean = Vec::with_capacity(CHUNK + 64);
    let per_chunk = CHUNK / SPS * bps;
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
        shifted.clear();
        offset.process(&wide, &mut shifted);
        clean.clear();
        drift.process(&shifted, &mut clean);
        std::thread::scope(|s| {
            for rx in &mut rxs {
                let clean = &clean;
                s.spawn(move || rx.process(clean));
            }
        });
    }
    let out = rxs
        .iter()
        .map(|rx| {
            let (m, slips) = rx.result();
            assert!(
                m.bits() >= BER_BITS,
                "Eb/N0 {}: only {} bits counted",
                rx.ebn0_db,
                m.bits()
            );
            (rx.ebn0_db, m.ber(), m.bits(), slips)
        })
        .collect();
    eprintln!(
        "order {order} differential {differential}: {:.1} s",
        start.elapsed().as_secs_f64()
    );
    out
}

/// Every point within `low(ebn0) <= BER <= high(ebn0)`, no slips; prints the figures.
fn check(
    name: &str,
    results: &[(f64, f64, u64, u64)],
    theory: fn(f64) -> f64,
    bounds: impl Fn(f64) -> (f64, f64),
) {
    let mut failures = Vec::new();
    for &(ebn0, ber, bits, slips) in results {
        let t = theory(ebn0);
        let (lo, hi) = bounds(ebn0);
        eprintln!(
            "{name} Eb/N0 {ebn0} dB: BER {ber:.4e} over {bits} bits, theory {t:.4e}, ratio \
             {:.3}, accepted {lo:.4e} .. {hi:.4e}, slips {slips}",
            ber / t
        );
        if !(ber >= lo && ber <= hi) || slips != 0 {
            failures.push(format!(
                "{name} Eb/N0 {ebn0} dB: BER {ber:.4e}, slips {slips}"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

fn factor(theory: fn(f64) -> f64, f: f64) -> impl Fn(f64) -> (f64, f64) {
    move |e| (theory(e) / f, theory(e) * f)
}

fn one_db(theory: fn(f64) -> f64) -> impl Fn(f64) -> (f64, f64) {
    move |e| (theory(e + 1.0), theory(e - 1.0))
}

#[test]
fn coherent_bpsk_ber_within_1_5_of_theory() {
    let r = ber_run(2, false, 553);
    check("bpsk", &r, coherent_theory, factor(coherent_theory, 1.5));
}

#[test]
fn coherent_qpsk_ber_within_1_5_of_theory() {
    let r = ber_run(4, false, 554);
    check("qpsk", &r, coherent_theory, factor(coherent_theory, 1.5));
}

#[test]
fn dbpsk_ber_within_1_db_of_theory() {
    let r = ber_run(2, true, 555);
    check("dbpsk", &r, dbpsk_theory, one_db(dbpsk_theory));
}

#[test]
fn dqpsk_ber_within_1_db_of_theory() {
    let r = ber_run(4, true, 556);
    check("dqpsk", &r, dqpsk_theory, one_db(dqpsk_theory));
}
