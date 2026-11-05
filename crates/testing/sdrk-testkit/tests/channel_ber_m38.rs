//! M38 tests 5 and 6: the PRBS bit error rate meter (injected errors, slip; seed 497), the
//! channel impairments (`Awgn::from_ebn0` over 10^6 samples, seed 498; `FrequencyOffset` and
//! `ClockDrift` against analytic tones, seed 499) and the BPSK/QPSK test signal.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]

use std::collections::BTreeSet;
use std::f64::consts::PI;

use num_complex::Complex64;
use sdrk_proto::prbs::{Poly, Prbs};
use sdrk_testkit::ber::{BerMeter, SLIP_WINDOW};
use sdrk_testkit::channel::{Awgn, ChannelChain, ClockDrift, FrequencyOffset, Impairment};
use sdrk_testkit::r#gen::{Psk, PskTest, Pulse, psk_test, pulse};
use sdrk_testkit::rng::Pcg32;

fn prbs_bits(poly: Poly, seed: u32, n: usize) -> Vec<u8> {
    let mut p = Prbs::new(poly, seed).unwrap();
    (0..n).map(|_| p.next_bit()).collect()
}

/// `count` distinct positions in `lo..hi`, at least `gap` apart, drawn with `rng`.
fn positions(rng: &mut Pcg32, lo: usize, hi: usize, count: usize, gap: usize) -> BTreeSet<usize> {
    assert!(count * 2 * gap < hi - lo, "too many positions for the gap");
    let mut set = BTreeSet::new();
    while set.len() < count {
        let p = lo + rng.below((hi - lo) as u32) as usize;
        if set.range(p.saturating_sub(gap)..p + gap).next().is_none() {
            set.insert(p);
        }
    }
    set
}

/// Random chunk lengths (1..=max) that add up to `total`.
fn chunks(rng: &mut Pcg32, total: usize, max: u32) -> Vec<usize> {
    let mut v = Vec::new();
    let mut left = total;
    while left > 0 {
        let n = (1 + rng.below(max) as usize).min(left);
        v.push(n);
        left -= n;
    }
    v
}

fn run_chunked(stage: &mut dyn Impairment, x: &[Complex64], plan: &[usize]) -> Vec<Complex64> {
    let mut out = Vec::new();
    let mut at = 0;
    for &n in plan {
        stage.process(&x[at..at + n], &mut out);
        at += n;
    }
    out
}

// ---------------------------------------------------------------------------------------
// BerMeter (test 5, seed 497)
// ---------------------------------------------------------------------------------------

#[test]
fn ber_meter_counts_injected_errors_exactly() {
    let mut rng = Pcg32::new(497, 1);
    for (poly, tx_seed, meter_seed) in [
        (Poly::Prbs15, 0x7fff, 0x7fff),
        (Poly::Prbs15, 0x1234, 1),
        (Poly::Prbs9, 0x0a5, 0x1ff),
    ] {
        let n = 200_000;
        let mut bits = prbs_bits(poly, tx_seed, n);
        // errors only after the acquisition (at most deg + 32 bits)
        let errs = positions(&mut rng, 100, n, 1000, 2);
        for &p in &errs {
            bits[p] ^= 1;
        }
        let mut meter = BerMeter::new(poly, meter_seed).unwrap();
        let mut locked_at = None;
        for (i, &b) in bits.iter().enumerate() {
            meter.push(b);
            if locked_at.is_none() && meter.is_locked() {
                locked_at = Some(i);
            }
        }
        let locked_at = locked_at.unwrap();
        let deg = poly.degree() as usize;
        if tx_seed == meter_seed {
            assert_eq!(locked_at, 31, "{poly:?}: lock after the first 32 bits");
        } else {
            assert!(locked_at < deg + 32, "{poly:?}: locked at {locked_at}");
        }
        assert_eq!(meter.bits(), (n - 1 - locked_at) as u64, "{poly:?}");
        assert_eq!(meter.errors(), 1000, "{poly:?}");
        assert_eq!(meter.slips(), 0, "{poly:?}");
        assert!((meter.ber() - 1000.0 / meter.bits() as f64).abs() < 1e-18);
        let (lo, hi) = meter.confidence_95();
        assert!(lo < meter.ber() && meter.ber() < hi);
    }
}

#[test]
fn ber_meter_detects_a_slip_and_resynchronizes() {
    let mut rng = Pcg32::new(497, 2);
    let poly = Poly::Prbs15;
    let n = 100_000;
    let slip_at = 40_000; // the received stream skips `skip` bits here
    let skip = 7;
    let mut src = prbs_bits(poly, 0x7fff, n + skip);
    let before = positions(&mut rng, 100, slip_at - 300, 100, 100);
    let after = positions(&mut rng, slip_at + 2000, n, 150, 100);
    for &p in &before {
        src[p] ^= 1;
    }
    for &p in &after {
        src[p + skip] ^= 1;
    }
    let rx: Vec<u8> = src[..slip_at]
        .iter()
        .chain(&src[slip_at + skip..])
        .copied()
        .collect();
    assert_eq!(rx.len(), n);

    let mut meter = BerMeter::new(poly, 0x7fff).unwrap();
    let mut at_slip = None;
    let mut lost_at = None;
    let mut relock = None;
    for (i, &b) in rx.iter().enumerate() {
        if i == slip_at {
            at_slip = Some((meter.bits(), meter.errors()));
        }
        meter.push(b);
        if i >= slip_at && lost_at.is_none() && !meter.is_locked() {
            lost_at = Some(i);
        }
        if lost_at.is_some() && relock.is_none() && meter.is_locked() {
            relock = Some((i, meter.bits(), meter.errors()));
        }
    }
    let (bits_before, errors_before) = at_slip.unwrap();
    assert_eq!(bits_before, (slip_at - 32) as u64);
    assert_eq!(errors_before, before.len() as u64);
    let lost_at = lost_at.unwrap();
    assert!(
        lost_at - slip_at <= 4 * SLIP_WINDOW as usize,
        "slip detected {} bits late",
        lost_at - slip_at
    );
    let (relock_at, bits_relock, errors_relock) = relock.unwrap();
    // acquisition: 15 bits to refill the register, then 32 correct predictions
    assert!(relock_at - lost_at <= 15 + 32, "{relock_at} {lost_at}");
    assert!(relock_at < slip_at + 2000);
    assert_eq!(meter.slips(), 1);
    assert_eq!(meter.bits() - bits_relock, (n - 1 - relock_at) as u64);
    assert_eq!(meter.errors() - errors_relock, after.len() as u64);
    // the slip window itself: about half of its bits are errors
    let slip_errors = errors_relock - errors_before;
    let slip_bits = bits_relock - bits_before;
    assert!(
        slip_errors >= 64 && slip_errors <= slip_bits,
        "{slip_errors} of {slip_bits}"
    );
}

#[test]
fn ber_meter_wilson_interval() {
    let mut meter = BerMeter::new(Poly::Prbs9, 1).unwrap();
    assert_eq!(meter.confidence_95(), (0.0, 1.0));
    let mut tx = Prbs::new(Poly::Prbs9, 1).unwrap();
    let total = 1_000_032;
    for i in 0..total {
        let b = tx.next_bit();
        // one error every 1000 counted bits
        meter.push(if i >= 32 && (i - 32) % 1000 == 999 {
            b ^ 1
        } else {
            b
        });
    }
    assert_eq!((meter.bits(), meter.errors()), (1_000_000, 1000));
    let (lo, hi) = meter.confidence_95();
    // Wilson interval of 1000 errors in 10^6 bits, computed with the closed form in f64
    // (z = 1.959963984540054) in Python: 0.0009399388437436418 .. 0.0010638949174321755
    assert!((lo - 9.399_388_437_436_418e-4).abs() < 1e-15, "{lo}");
    assert!((hi - 1.063_894_917_432_175_5e-3).abs() < 1e-15, "{hi}");
}

// ---------------------------------------------------------------------------------------
// Awgn (test 6, seed 498)
// ---------------------------------------------------------------------------------------

#[test]
fn awgn_from_ebn0_gives_the_measured_ebn0_within_0_1_db() {
    let n = 1_000_000;
    let sps = 4.1;
    let bits = prbs_bits(Poly::Prbs15, 0x7fff, (n as f64 / sps) as usize + 40);
    let mut sig = vec![Complex64::new(0.0, 0.0); n];
    psk_test(Psk::Bpsk, sps, 0.35, &bits, &mut sig);
    let power = sig.iter().map(Complex64::norm_sqr).sum::<f64>() / n as f64;
    for ebn0_db in [0.0, 6.0, 8.0, 20.0] {
        let mut awgn = Awgn::from_ebn0(ebn0_db, 1, sps, power, 498).unwrap();
        let sigma = awgn.sigma();
        let mut out = Vec::new();
        awgn.process(&sig, &mut out);
        assert_eq!(out.len(), n);
        let (mut nre, mut nim, mut cross, mut mean) = (0.0, 0.0, 0.0, Complex64::new(0.0, 0.0));
        for (y, x) in out.iter().zip(&sig) {
            let e = *y - *x;
            nre += e.re * e.re;
            nim += e.im * e.im;
            cross += e.re * e.im;
            mean += e;
        }
        let nf = n as f64;
        let noise = (nre + nim) / nf;
        let measured_db = 10.0 * (power * sps / (1.0 * noise)).log10();
        assert!(
            (measured_db - ebn0_db).abs() < 0.1,
            "eb/n0 {ebn0_db} dB: measured {measured_db} dB"
        );
        // both components carry sigma^2, independent and without a mean
        for v in [nre / nf, nim / nf] {
            assert!(
                (v / (sigma * sigma) - 1.0).abs() < 0.01,
                "{v} vs {}",
                sigma * sigma
            );
        }
        assert!((cross / nf).abs() < 0.01 * sigma * sigma);
        assert!(mean.norm_sqr().sqrt() / nf < 0.01 * sigma);
    }
}

#[test]
fn awgn_is_deterministic_and_chunk_invariant() {
    let x = vec![Complex64::new(0.25, -0.5); 5000];
    let mut a = Awgn::new(0.3, 498).unwrap();
    let mut whole = Vec::new();
    a.process(&x, &mut whole);
    let mut b = Awgn::new(0.3, 498).unwrap();
    let plan = chunks(&mut Pcg32::new(498, 3), x.len(), 700);
    assert_eq!(run_chunked(&mut b, &x, &plan), whole);
    let mut c = Awgn::new(0.3, 499).unwrap();
    let mut other = Vec::new();
    c.process(&x, &mut other);
    assert_ne!(other, whole);
}

#[test]
fn impairment_parameters_are_validated() {
    assert!(Awgn::new(f64::NAN, 1).is_err());
    assert!(Awgn::new(-0.1, 1).is_err());
    assert!(Awgn::from_ebn0(f64::INFINITY, 1, 4.0, 1.0, 1).is_err());
    assert!(Awgn::from_ebn0(6.0, 0, 4.0, 1.0, 1).is_err());
    assert!(Awgn::from_ebn0(6.0, 1, 0.0, 1.0, 1).is_err());
    assert!(Awgn::from_ebn0(6.0, 1, 4.0, -1.0, 1).is_err());
    assert!(FrequencyOffset::new(f64::NAN, 1.0).is_err());
    assert!(FrequencyOffset::new(1.0, -1.0).is_err());
    assert!(ClockDrift::new(f64::INFINITY).is_err());
    assert!(ClockDrift::new(1e5).is_err());
}

// ---------------------------------------------------------------------------------------
// FrequencyOffset and ClockDrift against analytic tones (test 6, seed 499)
// ---------------------------------------------------------------------------------------

fn tone(f: f64, phase: f64, from: i64, n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let a = 2.0 * PI * f * (from + i as i64) as f64 + phase;
            Complex64::new(a.cos(), a.sin())
        })
        .collect()
}

#[test]
fn frequency_offset_matches_analytic_tones() {
    let mut rng = Pcg32::new(499, 1);
    let fs = 48_000.0;
    let n = 100_000;
    for _ in 0..6 {
        let f0 = (rng.next_f64() - 0.5) * 0.8; // cycles per sample
        let phase = rng.next_f64() * 2.0 * PI;
        let offset_hz = (rng.next_f64() - 0.5) * 2000.0;
        let x = tone(f0, phase, 0, n);
        let mut whole = Vec::new();
        FrequencyOffset::new(offset_hz, fs)
            .unwrap()
            .process(&x, &mut whole);
        let expected = tone(f0 + offset_hz / fs, phase, 0, n);
        let err = whole
            .iter()
            .zip(&expected)
            .map(|(a, b)| (*a - *b).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            err < 1e-9,
            "f0 {f0}, offset {offset_hz} Hz: max error {err}"
        );
        let plan = chunks(&mut rng, n, 3000);
        let mut f = FrequencyOffset::new(offset_hz, fs).unwrap();
        assert_eq!(run_chunked(&mut f, &x, &plan), whole);
    }
}

/// Largest error of the drift output against the analytic tone `exp(j(2 pi f m r + phase))`
/// over the outputs whose window lies entirely inside the input.
fn drift_error(ppm: f64, f: f64, phase: f64, n: usize) -> (f64, usize) {
    let x = tone(f, phase, 0, n);
    let mut d = ClockDrift::new(ppm).unwrap();
    let r = d.ratio();
    let mut y = Vec::new();
    d.process(&x, &mut y);
    let mut worst = 0.0f64;
    let mut checked = 0;
    for (m, v) in y.iter().enumerate() {
        let p = m as f64 * r;
        if p < 32.0 {
            continue; // the window still reaches the zeros before index 0
        }
        let a = 2.0 * PI * f * p + phase;
        let e = (*v - Complex64::new(a.cos(), a.sin())).norm_sqr().sqrt();
        worst = worst.max(e);
        checked += 1;
    }
    (worst, checked)
}

#[test]
fn clock_drift_matches_analytic_tones() {
    let mut rng = Pcg32::new(499, 2);
    let n = 40_000;
    for ppm in [100.0, -100.0, 50.0, -40.0, 1000.0] {
        for f in [0.01, 0.1, 0.25, -0.3] {
            let phase = rng.next_f64() * 2.0 * PI;
            let (err, checked) = drift_error(ppm, f, phase, n);
            assert!(checked > n - n / 100, "{checked}");
            // 4-term Blackman–Harris window, 32 taps per side: measured below 1e-6 (-120 dB)
            // up to 0.3 cycles per sample
            assert!(err < 1e-5, "ppm {ppm}, f {f}: max error {err}");
        }
    }
}

#[test]
fn clock_drift_output_length_position_and_chunks() {
    let mut rng = Pcg32::new(499, 3);
    let n = 30_000;
    let x = tone(0.05, 0.3, 0, n);
    for ppm in [0.0, 100.0, -100.0] {
        let mut d = ClockDrift::new(ppm).unwrap();
        let r = d.ratio();
        let mut whole = Vec::new();
        d.process(&x, &mut whole);
        // output m exists exactly when floor(m r) + 32 < n
        let count = (0..n)
            .take_while(|&m: &usize| (m as f64 * r).floor() as usize + 32 < n)
            .count();
        assert_eq!(whole.len(), count, "ppm {ppm}");
        assert!((d.output_position(1000.0) - 1000.0 / r).abs() < 1e-12);
        let plan = chunks(&mut rng, n, 2500);
        let mut c = ClockDrift::new(ppm).unwrap();
        assert_eq!(run_chunked(&mut c, &x, &plan), whole, "ppm {ppm}");
        if ppm == 0.0 {
            assert_eq!(whole[..], x[..whole.len()]);
        }
    }
}

#[test]
fn clock_drift_moves_a_pulse_to_its_output_position() {
    // a raised-cosine pulse at the input position 5000.3 comes out at 5000.3 / r
    let mut g = PskTest::new(Psk::Bpsk, 8.0, 0.35);
    g.offset = 5000.3;
    let mut x = vec![Complex64::new(0.0, 0.0); 6000];
    g.generate(&[1], 0, &mut x);
    for ppm in [1000.0, -1000.0] {
        let mut d = ClockDrift::new(ppm).unwrap();
        let mut y = Vec::new();
        d.process(&x, &mut y);
        let q = d.output_position(g.offset);
        // the peak and the symmetric neighbours around the expected position
        let m = q.floor() as usize;
        let mu = q - m as f64;
        for k in 0..4 {
            let expected = pulse(
                Pulse::RaisedCosine,
                0.35,
                ((m + k) as f64 - q) * d.ratio() / 8.0,
            );
            assert!(
                (y[m + k].re - expected).abs() < 1e-4,
                "ppm {ppm}, k {k}, mu {mu}"
            );
        }
    }
}

#[test]
fn channel_chain_applies_its_stages_in_order() {
    let x = tone(0.02, 0.0, 0, 8000);
    let mut chain = ChannelChain::new()
        .with(ClockDrift::new(80.0).unwrap())
        .with(FrequencyOffset::new(30.0, 1000.0).unwrap())
        .with(Awgn::new(0.01, 499).unwrap());
    let mut by_chain = Vec::new();
    let plan = chunks(&mut Pcg32::new(499, 4), x.len(), 900);
    let got = run_chunked(&mut chain, &x, &plan);
    by_chain.extend(got);

    let mut a = Vec::new();
    ClockDrift::new(80.0).unwrap().process(&x, &mut a);
    let mut b = Vec::new();
    FrequencyOffset::new(30.0, 1000.0)
        .unwrap()
        .process(&a, &mut b);
    let mut c = Vec::new();
    Awgn::new(0.01, 499).unwrap().process(&b, &mut c);
    assert_eq!(by_chain, c);

    let mut empty = ChannelChain::new();
    let mut out = Vec::new();
    empty.process(&x[..10], &mut out);
    assert_eq!(out, x[..10]);
}

// ---------------------------------------------------------------------------------------
// PSK test signal
// ---------------------------------------------------------------------------------------

#[test]
fn psk_test_hits_its_symbols_at_integer_sps() {
    let bits = prbs_bits(Poly::Prbs9, 0x1ff, 400);
    for modulation in [Psk::Bpsk, Psk::Qpsk] {
        let g = PskTest::new(modulation, 4.0, 0.35);
        let mut s = vec![Complex64::new(0.0, 0.0); g.samples(&bits)];
        g.generate(&bits, 0, &mut s);
        for k in 0..g.symbols(&bits) {
            let at = g.symbol_position(k) as usize;
            assert!(
                (s[at] - g.symbol(&bits, k)).norm_sqr().sqrt() < 1e-12,
                "{modulation:?} {k}"
            );
        }
    }
}

#[test]
fn psk_test_symbol_mapping() {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let g = PskTest::new(Psk::Bpsk, 2.0, 0.5);
    assert_eq!(g.symbol(&[0, 1, 7], 0), Complex64::new(-1.0, 0.0));
    assert_eq!(g.symbol(&[0, 1, 7], 1), Complex64::new(1.0, 0.0));
    assert_eq!(g.symbol(&[0, 1, 7], 2), Complex64::new(1.0, 0.0));
    assert_eq!(g.symbol(&[0, 1, 7], 3), Complex64::new(0.0, 0.0));
    let q = PskTest::new(Psk::Qpsk, 2.0, 0.5);
    let bits = [0, 0, 0, 1, 1, 0, 1, 1];
    let expected = [(-h, -h), (-h, h), (h, -h), (h, h)];
    for (k, (re, im)) in expected.into_iter().enumerate() {
        assert_eq!(q.symbol(&bits, k), Complex64::new(re, im), "{k}");
    }
}

#[test]
fn psk_test_power_and_chunks() {
    let bits = prbs_bits(Poly::Prbs15, 0x7fff, 20_000);
    for (modulation, pulse_shape, beta, power) in [
        (Psk::Bpsk, Pulse::RaisedCosine, 0.35, 1.0 - 0.35 / 4.0),
        (Psk::Qpsk, Pulse::RaisedCosine, 0.5, 1.0 - 0.5 / 4.0),
        (Psk::Bpsk, Pulse::RootRaisedCosine, 0.35, 1.0),
    ] {
        let mut g = PskTest::new(modulation, 4.1, beta);
        g.pulse = pulse_shape;
        g.offset = 0.37;
        let total = g.samples(&bits);
        let mut s = vec![Complex64::new(0.0, 0.0); total];
        g.generate(&bits, 0, &mut s);
        // mean power over the steady part (energy of the pulse per symbol)
        let lo = 400;
        let hi = total - 400;
        let p = s[lo..hi].iter().map(Complex64::norm_sqr).sum::<f64>() / (hi - lo) as f64;
        assert!(
            (p / power - 1.0).abs() < 0.03,
            "{modulation:?} {pulse_shape:?}: {p}"
        );
        let mut c = vec![Complex64::new(0.0, 0.0); 1234];
        g.generate(&bits, 5000, &mut c);
        assert_eq!(c[..], s[5000..6234]);
    }
}

#[test]
fn root_raised_cosine_convolved_with_itself_is_raised_cosine() {
    let sps = 8.0;
    let beta = 0.35;
    let half = 16 * 8;
    let h: Vec<f64> = (-half..=half)
        .map(|n| pulse(Pulse::RootRaisedCosine, beta, n as f64 / sps))
        .collect();
    for lag in [0i64, 4, 8, 12, 16, 24] {
        let mut acc = 0.0;
        for (j, a) in h.iter().enumerate() {
            let k = j as i64 - lag;
            if k >= 0 && (k as usize) < h.len() {
                acc += a * h[k as usize];
            }
        }
        let expected = pulse(Pulse::RaisedCosine, beta, lag as f64 / sps);
        assert!(
            (acc / sps - expected).abs() < 3e-3,
            "lag {lag}: {} vs {expected}",
            acc / sps
        );
    }
    // the singular points take their limits continuously
    for (shape, t) in [
        (Pulse::RaisedCosine, 0.5 / beta),
        (Pulse::RootRaisedCosine, 0.25 / beta),
    ] {
        let at = pulse(shape, beta, t);
        let near = pulse(shape, beta, t + 1e-6);
        assert!((at - near).abs() < 1e-5, "{shape:?}: {at} vs {near}");
    }
}

#[test]
fn public_types_are_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Awgn>();
    assert_send::<FrequencyOffset>();
    assert_send::<ClockDrift>();
    assert_send::<ChannelChain>();
    assert_send::<BerMeter>();
    assert_send::<PskTest>();
    assert_send::<Psk>();
    assert_send::<Pulse>();
}
