//! The loopback channel simulator.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::result_large_err,
    clippy::cast_lossless,
    clippy::many_single_char_names
)]

use sdrk_device::{BlockFlags, BlockMeta, Rate, RxStream, SampleIndex, TxStream, formats};
use sdrk_testkit::channel::{Loopback, LoopbackParams, LoopbackRx, LoopbackTx};
use sdrk_testkit::encode::{decode_reference, encode};
use sdrk_testkit::r#gen;
use sdrk_testkit::meters::estimate_frequency;

use num_complex::Complex64;

const RATE: u64 = 1_000_000;

/// `Complex::norm` needs `std` or `libm` features of num-complex, which the testkit does not enable.
trait Mag {
    fn norm(&self) -> f64;
}

impl Mag for Complex64 {
    fn norm(&self) -> f64 {
        self.norm_sqr().sqrt()
    }
}

fn params() -> LoopbackParams {
    LoopbackParams::new(Rate::hz(RATE).unwrap(), formats::CF32)
}

fn tx_complex(tx: &mut LoopbackTx, x: &[Complex64], meta: &BlockMeta) {
    let mut bytes = vec![0u8; x.len() * 8];
    encode(&formats::CF32, x, &mut bytes).unwrap();
    assert_eq!(tx.write(&bytes, meta).unwrap(), bytes.len());
}

fn rx_complex(rx: &mut LoopbackRx, n: usize, block: usize) -> Vec<Complex64> {
    let mut out = Vec::with_capacity(n);
    let mut bytes = vec![0u8; block * 8];
    while out.len() < n {
        let want = (n - out.len()).min(block);
        let (got, _) = rx.read(&mut bytes[..want * 8]).unwrap();
        let mut samples = vec![Complex64::new(0.0, 0.0); got / 8];
        decode_reference(&formats::CF32, &bytes[..got], &mut samples).unwrap();
        out.extend(samples);
    }
    out
}

/// Peak position of `|y|` by parabolic interpolation of the logarithm through the maximum and
/// its neighbours (exact for a Gaussian pulse).
fn peak_position(y: &[Complex64]) -> f64 {
    let a: Vec<f64> = y.iter().map(|v| v.norm().ln()).collect();
    let (k, _) = a
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .unwrap();
    let (l, c, r) = (a[k - 1], a[k], a[k + 1]);
    k as f64 + 0.5 * (l - r) / (l - 2.0 * c + r)
}

/// A unit impulse is a poor test signal for a position estimate by parabolic interpolation: the
/// response around its peak is a sinc, whose main lobe is not a parabola (the estimate is
/// biased by up to 0.1 sample). A Gaussian pulse (sigma 4 samples, band-limited far below the
/// interpolator's pass band) is the impulse response test with an unbiased estimator.
#[test]
fn delay_accuracy_with_a_pulse() {
    let center = 40.0;
    for delay in [100.0, 100.1, 100.25, 100.37, 100.5, 100.73, 100.99, 1234.37] {
        let mut p = params();
        p.delay_samples = delay;
        let (mut tx, mut rx) = Loopback::new(p).unwrap().split();
        let x: Vec<Complex64> = (0..80)
            .map(|n| Complex64::new((-((f64::from(n) - center).powi(2)) / 32.0).exp(), 0.0))
            .collect();
        tx_complex(&mut tx, &x, &BlockMeta::default());
        let y = rx_complex(&mut rx, 1400, 333);
        let peak = peak_position(&y);
        assert!(
            (peak - (center + delay)).abs() < 0.01,
            "delay {delay}: peak at {peak}, expected {}",
            center + delay
        );
    }
}

fn i0(x: f64) -> f64 {
    let (mut term, mut sum) = (1.0, 1.0);
    for k in 1..60 {
        term *= (x / 2.0) * (x / 2.0) / f64::from(k * k);
        sum += term;
    }
    sum
}

/// The response to a unit impulse is the Kaiser(8)-windowed sinc, tap by tap.
#[test]
fn an_impulse_returns_the_windowed_sinc_taps() {
    let (delay, frac) = (50.3_f64, 0.3_f64);
    let mut p = params();
    p.delay_samples = delay;
    let (mut tx, mut rx) = Loopback::new(p).unwrap().split();
    let mut x = vec![Complex64::new(0.0, 0.0); 4];
    x[1] = Complex64::new(1.0, 0.0);
    tx_complex(&mut tx, &x, &BlockMeta::default());
    let y = rx_complex(&mut rx, 200, 200);
    for j in -31i32..=31 {
        let u = f64::from(j) - frac;
        let w = i0(8.0 * (1.0 - (u / 32.0).powi(2)).sqrt()) / i0(8.0);
        let sinc = if u == 0.0 {
            1.0
        } else {
            (std::f64::consts::PI * u).sin() / (std::f64::consts::PI * u)
        };
        // y[n] = g[j] x[n - 50 - j] with x[1] = 1  ->  n = 51 + j
        let got = y[(51 + j) as usize].re;
        assert!(
            (got - sinc * w).abs() < 1e-6,
            "tap {j}: {got} vs {}",
            sinc * w
        );
    }
    assert_eq!(y[51 - 32].re, 0.0);
    assert_eq!(y[51 + 32].re, 0.0);
}

#[test]
fn an_integer_delay_is_an_exact_shift() {
    let mut p = params();
    p.delay_samples = 5.0;
    let (mut tx, mut rx) = Loopback::new(p).unwrap().split();
    let mut x = vec![Complex64::new(0.0, 0.0); 50];
    r#gen::tone(&mut x, 0.0123, 0.7, 0.3, 0);
    tx_complex(&mut tx, &x, &BlockMeta::default());
    let y = rx_complex(&mut rx, 60, 7);
    for n in 0..60 {
        let want = if (5..55).contains(&n) {
            x[n - 5]
        } else {
            Complex64::new(0.0, 0.0)
        };
        // CF32 stores f32: compare with the rounded input
        let want = Complex64::new(f64::from(want.re as f32), f64::from(want.im as f32));
        assert_eq!(y[n], want, "sample {n}");
    }
}

#[test]
fn attenuation_and_frequency_offset() {
    let mut p = params();
    p.attenuation_db = 6.0;
    p.freq_offset_hz = 12_345.0;
    let (mut tx, mut rx) = Loopback::new(p).unwrap().split();
    let mut x = vec![Complex64::new(0.0, 0.0); 20_000];
    r#gen::tone(&mut x, 0.01, 0.8, 0.0, 0);
    tx_complex(&mut tx, &x, &BlockMeta::default());
    let y = rx_complex(&mut rx, 20_000, 4096);
    let amp = y[1000..19000].iter().map(Mag::norm).sum::<f64>() / 18_000.0;
    let want = 0.8 * 10f64.powf(-6.0 / 20.0);
    assert!((amp - want).abs() < 1e-6, "amplitude {amp} vs {want}");
    let f = estimate_frequency(&y[1000..19000], RATE as f64);
    assert!((f - (10_000.0 + 12_345.0)).abs() < 1.0, "frequency {f}");
    // the phase follows the absolute index: sample 0 is not rotated
    assert!((y[0].re - 0.8 * want / 0.8).abs() < 1e-6);
}

#[test]
fn noise_level_matches_sigma() {
    for sigma in [0.001, 0.01, 0.1] {
        let mut p = params();
        p.noise_sigma = sigma;
        p.seed = 77;
        let (_tx, mut rx) = Loopback::new(p).unwrap().split();
        let y = rx_complex(&mut rx, 200_000, 4096);
        let power = y.iter().map(Complex64::norm_sqr).sum::<f64>() / y.len() as f64;
        let db = 10.0 * (power / (2.0 * sigma * sigma)).log10();
        assert!(db.abs() < 0.2, "sigma {sigma}: {db} dB");
        let mean = y.iter().map(|v| v.re).sum::<f64>() / y.len() as f64;
        assert!(mean.abs() < 5.0 * sigma / (y.len() as f64).sqrt());
    }
}

#[test]
fn noise_depends_on_the_seed_and_not_on_the_chunking() {
    let mut p = params();
    p.noise_sigma = 0.05;
    p.seed = 5;
    p.delay_samples = 3.4;
    p.freq_offset_hz = 777.0;
    p.attenuation_db = 3.0;
    let run = |seed: u64, block: usize| {
        let mut q = p;
        q.seed = seed;
        let (mut tx, mut rx) = Loopback::new(q).unwrap().split();
        let mut x = vec![Complex64::new(0.0, 0.0); 3000];
        r#gen::tone(&mut x, 0.02, 0.5, 0.0, 0);
        tx_complex(&mut tx, &x, &BlockMeta::default());
        let mut bytes = Vec::new();
        let mut buf = vec![0u8; block * 8];
        while bytes.len() < 5000 * 8 {
            let (n, _) = rx.read(&mut buf).unwrap();
            bytes.extend_from_slice(&buf[..n]);
        }
        bytes.truncate(5000 * 8);
        bytes
    };
    let reference = run(5, 4096);
    for block in [1, 7, 64, 1000] {
        assert_eq!(run(5, block), reference, "block {block}");
    }
    assert_ne!(run(6, 4096), reference);
}

#[test]
fn bursts_are_placed_by_their_transmit_time() {
    let mut p = params();
    p.t0_ns = 1_700_000_000_000_000_000;
    p.delay_samples = 4.0;
    let (mut tx, mut rx) = Loopback::new(p).unwrap().split();
    let ones = vec![Complex64::new(0.5, 0.0); 100];
    // 1.5 ms after t0 at 1 MS/s: index 1500 (+ delay 4)
    let burst = BlockMeta {
        flags: BlockFlags::BURST_START,
        tx_time_ns: Some(p.t0_ns + 1_500_000),
        ..BlockMeta::default()
    };
    tx_complex(&mut tx, &ones[..60], &burst);
    tx_complex(&mut tx, &ones[60..], &BlockMeta::default());
    // a second burst; 2.0004 ms rounds to index 2000 (0.4 sample)
    let second = BlockMeta {
        flags: BlockFlags::BURST_START.union(BlockFlags::BURST_END),
        tx_time_ns: Some(p.t0_ns + 3_000_400),
        ..BlockMeta::default()
    };
    tx_complex(&mut tx, &ones[..10], &second);
    let y = rx_complex(&mut rx, 3100, 512);
    let active: Vec<usize> = (0..y.len()).filter(|&n| y[n].norm() > 0.25).collect();
    assert_eq!(active[0], 1504);
    assert_eq!(*active.iter().find(|&&n| n > 1700).unwrap(), 3004);
    assert_eq!(active.iter().filter(|&&n| n < 2000).count(), 100);
    assert_eq!(active.len(), 110);
    assert_eq!(y[1503].norm(), 0.0);
    assert_eq!(y[1604].norm(), 0.0);
}

#[test]
fn continuous_transmission_starts_at_index_zero_and_continues() {
    let (mut tx, mut rx) = Loopback::new(params()).unwrap().split();
    let a = vec![Complex64::new(0.25, 0.0); 10];
    let b = vec![Complex64::new(-0.25, 0.0); 10];
    tx_complex(&mut tx, &a, &BlockMeta::default());
    tx_complex(&mut tx, &b, &BlockMeta::default());
    let y = rx_complex(&mut rx, 30, 30);
    assert!(y[..10].iter().all(|v| v.re == 0.25));
    assert!(y[10..20].iter().all(|v| v.re == -0.25));
    assert!(y[20..].iter().all(|v| v.re == 0.0));
}

#[test]
fn a_burst_before_t0_is_late_and_dropped() {
    let mut p = params();
    p.t0_ns = 10_000_000;
    let (mut tx, mut rx) = Loopback::new(p).unwrap().split();
    let ones = vec![Complex64::new(0.5, 0.0); 20];
    let burst = BlockMeta {
        flags: BlockFlags::BURST_START.union(BlockFlags::BURST_END),
        tx_time_ns: Some(5_000_000),
        ..BlockMeta::default()
    };
    tx_complex(&mut tx, &ones, &burst);
    let status = tx.poll_status().unwrap().unwrap();
    assert!(status.flags.contains(BlockFlags::LATE));
    assert_eq!(tx.poll_status().unwrap(), None);
    let y = rx_complex(&mut rx, 100, 100);
    assert!(y.iter().all(|v| v.norm() == 0.0));
}

#[test]
fn the_first_block_carries_the_anchor_and_indices_continue() {
    let mut p = params();
    p.t0_ns = 42;
    p.max_block = 100;
    let (_tx, mut rx) = Loopback::new(p).unwrap().split();
    let mut buf = vec![0u8; 1000 * 8];
    let (n, m0) = rx.read(&mut buf).unwrap();
    assert_eq!(n, 100 * 8);
    let a = m0.anchor.unwrap();
    assert_eq!((a.index, a.unix_ns), (SampleIndex(0), 42));
    let (_, m1) = rx.read(&mut buf).unwrap();
    assert_eq!(m1.start, SampleIndex(100));
    assert!(m1.anchor.is_none());
    assert_eq!(rx.next_index(), 200);
    assert_eq!(rx.sample_rate(), Rate::hz(RATE).unwrap());
    assert_eq!(rx.format(), formats::CF32);
}

#[test]
fn the_receiver_saturates_in_integer_formats() {
    let mut p = LoopbackParams::new(Rate::hz(RATE).unwrap(), formats::CS8);
    p.attenuation_db = -20.0; // gain 10
    let (mut tx, mut rx) = Loopback::new(p).unwrap().split();
    let mut bytes = vec![0u8; 4];
    encode(
        &formats::CS8,
        &[Complex64::new(0.5, -0.5), Complex64::new(0.0, 0.0)],
        &mut bytes,
    )
    .unwrap();
    tx.write(&bytes, &BlockMeta::default()).unwrap();
    let mut out = [0u8; 4];
    rx.read(&mut out).unwrap();
    assert_eq!(out, [127, 128, 0, 0]);
}

#[test]
fn loopback_runs_with_the_conformance_checks() {
    let mut p = params();
    p.delay_samples = 2.5;
    p.noise_sigma = 0.01;
    p.max_block = 100;
    let (mut tx, mut rx) = Loopback::new(p).unwrap().split();
    let r = sdrk_testkit::conformance::check_rx(&mut rx, 30, 2000).unwrap();
    assert_eq!((r.blocks, r.anchors), (30, 1));
    let r = sdrk_testkit::conformance::check_tx(&mut tx, 5, 400).unwrap();
    assert_eq!(r.blocks, 5);
}

#[test]
fn invalid_parameters_are_rejected() {
    let bad = |f: &dyn Fn(&mut LoopbackParams)| {
        let mut p = params();
        f(&mut p);
        Loopback::new(p).is_err()
    };
    assert!(bad(&|p| p.delay_samples = -1.0));
    assert!(bad(&|p| p.delay_samples = f64::NAN));
    assert!(bad(&|p| p.attenuation_db = f64::INFINITY));
    assert!(bad(&|p| p.freq_offset_hz = f64::NAN));
    assert!(bad(&|p| p.noise_sigma = -0.1));
    assert!(bad(&|p| p.max_block = 0));
    assert!(bad(&|p| p.format = formats::PACKED10));
    let (mut tx, mut rx) = Loopback::new(params()).unwrap().split();
    assert!(tx.write(&[0; 5], &BlockMeta::default()).is_err());
    assert!(rx.read(&mut [0u8; 4]).is_err());
    assert_eq!(tx.write(&[], &BlockMeta::default()).unwrap(), 0);
}

#[test]
fn public_types_are_send() {
    fn send<T: Send>() {}
    send::<Loopback>();
    send::<LoopbackTx>();
    send::<LoopbackRx>();
    send::<LoopbackParams>();
}
