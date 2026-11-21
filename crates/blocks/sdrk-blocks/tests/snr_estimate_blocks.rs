//! The building block `snr_estimate` (M44.4 §1): bias and spread against the truth per
//! constellation and Es/N0, over white Gaussian noise and through `Multipath::mild` after an
//! equalizer (seed 900, test 1); the estimate independent of the cut of the input with 20 cut
//! plans (seed 902); extreme inputs; the amplitude does not matter.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::print_stderr,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

mod common;

use common::{rate, run_block, twenty_plans};
use num_complex::Complex64;
use sdrk::C32;
use sdrk_blocks::{
    Block, ParamValue, PulseMatchBlock, PulseMatchConfig, PulseShapeBlock, PulseShapeConfig,
    SNR_ESTIMATE_FLOOR_DB, SnrEstimateBlock, SnrEstimateConfig, SymbolMapConstellation,
};
use sdrk_core::kernels::constellation::Qam;
use sdrk_testkit::channel::{Impairment, Multipath};
use sdrk_testkit::{Pcg32, VecStorage};

/// The seed of test 1 (bias and spread).
const SEED: u64 = 900;
/// The seed of the cut plans and their signals.
const SEED_CHUNKS: u64 = 902;
/// The default window.
const WINDOW: usize = 4096;
/// The estimates per point of test 1 over white Gaussian noise (after the first window, which
/// only measures the amplitude).
const AWGN_ESTIMATES: usize = 24;
/// The estimates per point of test 1 through `mild`.
const MILD_ESTIMATES: usize = 24;
/// Es/N0 of test 1 in dB.
const ES_N0: [f64; 9] = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0];

fn constellation(q: Qam) -> SymbolMapConstellation {
    match q {
        Qam::Qpsk => SymbolMapConstellation::Qpsk,
        Qam::Qam16 => SymbolMapConstellation::Qam16,
        Qam::Qam64 => SymbolMapConstellation::Qam64,
        Qam::Qam256 => SymbolMapConstellation::Qam256,
        _ => SymbolMapConstellation::Qam1024,
    }
}

fn block(q: Qam, window: usize) -> SnrEstimateBlock<VecStorage> {
    let cfg = SnrEstimateConfig {
        constellation: constellation(q),
        window,
    };
    SnrEstimateBlock::new(&cfg, rate(10_000), &mut VecStorage).unwrap()
}

/// `n` random unit-power symbols of `q`.
fn symbols(q: Qam, n: usize, rng: &mut Pcg32) -> Vec<Complex64> {
    (0..n)
        .map(|_| {
            let p = q.point(rng.below(q.points() as u32));
            Complex64::new(f64::from(p.re), f64::from(p.im))
        })
        .collect()
}

/// Complex Gaussian noise of power `power` added to `x`.
fn add_noise(x: &mut [Complex64], power: f64, rng: &mut Pcg32) {
    let sd = (power / 2.0).sqrt();
    for v in x {
        *v += Complex64::new(rng.normal() * sd, rng.normal() * sd);
    }
}

fn to_c32(x: &[Complex64], gain: f64) -> Vec<C32> {
    x.iter()
        .map(|v| C32::new((v.re * gain) as f32, (v.im * gain) as f32))
        .collect()
}

/// The estimate after every window, fed one window per call.
fn estimates(est: &mut SnrEstimateBlock<VecStorage>, x: &[C32], window: usize) -> Vec<f64> {
    let mut out = vec![C32::default(); window];
    let mut got = Vec::new();
    for piece in x.chunks(window) {
        let p = est.process(piece, &mut out[..piece.len()]).unwrap();
        assert_eq!(p.consumed, piece.len());
        if piece.len() == window {
            got.push(est.snr_db());
        }
    }
    got
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    (m, var.sqrt())
}

/// The bound of test 1 on the spread (standard deviation) of the estimates of one window of
/// 4096 symbols, in dB: 0.5 dB where the constellation works and its decisions are reliable,
/// 1 dB where the reading of the decisions is flat and its inverse spreads (256-QAM at 15 to 20
/// dB, 1024-QAM at 15 to 25 dB, D2501).
fn spread_bound(q: Qam, es_n0: f64) -> f64 {
    match q {
        Qam::Qam256 if (15.0..=20.0).contains(&es_n0) => 1.0,
        Qam::Qam1024 if (15.0..=25.0).contains(&es_n0) => 1.0,
        _ => 0.5,
    }
}

/// The bound of test 1 on the bias (the mean of the estimates less the truth) in dB: 0.25 dB,
/// and 0.6 dB where the spread may reach 1 dB (three standard errors of the mean of 24
/// estimates, D2501).
fn bias_bound(q: Qam, es_n0: f64) -> f64 {
    if spread_bound(q, es_n0) > 0.5 {
        0.6
    } else {
        0.25
    }
}

#[test]
fn test1_bias_and_spread_over_white_gaussian_noise() {
    let mut rows = Vec::new();
    for (k, q) in Qam::ALL.into_iter().enumerate() {
        for (j, &es_n0) in ES_N0.iter().enumerate() {
            let mut rng = Pcg32::new(SEED, (k * 16 + j) as u64);
            let n = (AWGN_ESTIMATES + 1) * WINDOW;
            let mut x = symbols(q, n, &mut rng);
            add_noise(&mut x, 10f64.powf(-es_n0 / 10.0), &mut rng);
            // Any amplitude: the block measures it.
            let x = to_c32(&x, 0.37);
            let got = estimates(&mut block(q, WINDOW), &x, WINDOW);
            assert!(
                got[0].is_nan(),
                "{q:?} {es_n0}: the first window gives no estimate"
            );
            let (m, sd) = mean_sd(&got[1..]);
            rows.push(format!(
                "AWGN {q:?} Es/N0 {es_n0:4.1} dB: bias {:+.3} dB, sd {sd:.3} dB",
                m - es_n0
            ));
            assert!(
                (m - es_n0).abs() <= bias_bound(q, es_n0),
                "{q:?} at {es_n0} dB: bias {}",
                m - es_n0
            );
            assert!(sd <= spread_bound(q, es_n0), "{q:?} at {es_n0} dB: sd {sd}");
        }
    }
    for r in rows {
        eprintln!("{r}");
    }
}

/// Solves `a w = b` (complex, `a` square) by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<Complex64>>, mut b: Vec<Complex64>) -> Vec<Complex64> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| a[i][c].norm_sqr().total_cmp(&a[j][c].norm_sqr()))
            .unwrap();
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..n {
                let v = a[c][k];
                a[r][k] -= f * v;
            }
            let v = b[c];
            b[r] -= f * v;
        }
    }
    let mut w = vec![Complex64::new(0.0, 0.0); n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for k in r + 1..n {
            s -= a[r][k] * w[k];
        }
        w[r] = s / a[r][r];
    }
    w
}

/// The taps of the equalizer of the mild channel.
const EQ_TAPS: usize = 15;
/// Samples per symbol of the mild channel.
const SPS: usize = 4;
/// Symbols that train the least-squares equalizer before the estimate.
const TRAIN: usize = 8192;

/// The symbols `s` through the root-raised-cosine pulse (4 samples per symbol), `mild`, noise at
/// `es_n0` of the received signal and the matched filter, sampled at the symbol centres and
/// aligned with `s` (the delay found by correlation): one value per symbol.
fn through_mild(s: &[Complex64], es_n0: f64, rng: &mut Pcg32) -> Vec<Complex64> {
    let r = rate((SPS * 10_000) as u64);
    let shape = PulseShapeConfig {
        samples_per_symbol: SPS,
        ..PulseShapeConfig::default()
    };
    let mut tx = PulseShapeBlock::new(&shape, rate(10_000), &mut VecStorage).unwrap();
    let samples = run_block(&mut tx, &to_c32(s, 1.0), None);
    let x: Vec<Complex64> = samples
        .iter()
        .map(|v| Complex64::new(f64::from(v.re), f64::from(v.im)))
        .collect();
    let mut ch = Multipath::mild(SPS as f64).unwrap();
    let mut y = Vec::new();
    ch.process(&x, &mut y);
    // Es/N0 of the received signal: the noise power per sample is the symbol energy (the power
    // per sample times the samples per symbol) over Es/N0.
    let power = y.iter().map(Complex64::norm_sqr).sum::<f64>() / y.len() as f64;
    add_noise(&mut y, power * SPS as f64 / 10f64.powf(es_n0 / 10.0), rng);
    let mf = PulseMatchConfig {
        symbol_rate_hz: 10_000.0,
        ..PulseMatchConfig::default()
    };
    let mut rx = PulseMatchBlock::new(&mf, r, &mut VecStorage).unwrap();
    let z = run_block(&mut rx, &to_c32(&y, 1.0), None);
    let z: Vec<Complex64> = z
        .iter()
        .map(|v| Complex64::new(f64::from(v.re), f64::from(v.im)))
        .collect();
    // The sample offset with the largest correlation with the symbols.
    let corr = |off: usize| {
        (0..4000)
            .map(|i| z.get(off + SPS * i).copied().unwrap_or_default() * s[i].conj())
            .sum::<Complex64>()
            .norm_sqr()
    };
    let off = (0..SPS * 40)
        .max_by(|&a, &b| corr(a).total_cmp(&corr(b)))
        .unwrap();
    (0..s.len())
        .map(|i| z.get(off + SPS * i).copied().unwrap_or_default())
        .collect()
}

/// The least-squares linear equalizer of `EQ_TAPS` taps trained on the symbols `TRAIN` and
/// applied to everything; output `n` estimates `s[n]`.
fn equalize(y: &[Complex64], s: &[Complex64]) -> Vec<Complex64> {
    let half = EQ_TAPS / 2;
    let window = |n: usize| -> Vec<Complex64> {
        (0..EQ_TAPS)
            .map(|k| {
                (n + k)
                    .checked_sub(half)
                    .and_then(|i| y.get(i))
                    .copied()
                    .unwrap_or_default()
            })
            .collect()
    };
    let mut a = vec![vec![Complex64::new(0.0, 0.0); EQ_TAPS]; EQ_TAPS];
    let mut b = vec![Complex64::new(0.0, 0.0); EQ_TAPS];
    for n in 100..TRAIN {
        let w = window(n);
        for r in 0..EQ_TAPS {
            for c in 0..EQ_TAPS {
                a[r][c] += w[r].conj() * w[c];
            }
            b[r] += w[r].conj() * s[n];
        }
    }
    let taps = solve(a, b);
    (0..y.len())
        .map(|n| window(n).iter().zip(&taps).map(|(x, t)| x * t).sum())
        .collect()
}

/// The signal-to-noise ratio of `e` against the sent `s` after the least-squares gain, in dB:
/// what the estimate measures.
fn genie_db(e: &[Complex64], s: &[Complex64]) -> f64 {
    let ss: f64 = s.iter().map(Complex64::norm_sqr).sum();
    let g = e.iter().zip(s).map(|(e, s)| (e * s.conj()).re).sum::<f64>() / ss;
    let err: f64 = e.iter().zip(s).map(|(e, s)| (e - s * g).norm_sqr()).sum();
    10.0 * (g * g * ss / err).log10()
}

#[test]
fn test1_bias_and_spread_through_mild_after_the_equalizer() {
    let mut rows = Vec::new();
    let mut failed = Vec::new();
    for (k, q) in Qam::ALL.into_iter().enumerate() {
        for (j, &es_n0) in ES_N0.iter().enumerate() {
            let mut rng = Pcg32::new(SEED, 1000 + (k * 16 + j) as u64);
            // The training symbols, the window of the amplitude, the estimates and the tail of
            // the filters.
            let n = TRAIN + (MILD_ESTIMATES + 1) * WINDOW + 64;
            let s = symbols(q, n, &mut rng);
            let y = through_mild(&s, es_n0, &mut rng);
            let e = equalize(&y, &s);
            let (e, s) = (
                &e[TRAIN..TRAIN + (MILD_ESTIMATES + 1) * WINDOW],
                &s[TRAIN..],
            );
            let got = estimates(&mut block(q, WINDOW), &to_c32(e, 1.0), WINDOW);
            let mut diff = Vec::new();
            let mut genie = Vec::new();
            for w in 1..=MILD_ESTIMATES {
                let r = w * WINDOW..(w + 1) * WINDOW;
                let g = genie_db(&e[r.clone()], &s[r]);
                genie.push(g);
                diff.push(got[w] - g);
            }
            let (m, sd) = mean_sd(&diff);
            let (g, _) = mean_sd(&genie);
            rows.push(format!(
                "mild {q:?} Es/N0 {es_n0:4.1} dB: after the equalizer {g:.2} dB, bias {m:+.3} dB, \
                 sd {sd:.3} dB"
            ));
            if m.abs() > bias_bound(q, es_n0) || sd > spread_bound(q, es_n0) {
                failed.push(format!("{q:?} at {es_n0} dB: bias {m}, sd {sd}"));
            }
        }
    }
    for r in rows {
        eprintln!("{r}");
    }
    assert!(failed.is_empty(), "{failed:?}");
}

/// A 16-QAM and a 1024-QAM signal of seed 902 and the estimates of a window of 1000 symbols,
/// fed in 20 cut plans: the output is the input bit for bit, and after every call the estimate
/// is the one of the windows completed so far, bit for bit.
#[test]
fn the_estimate_does_not_depend_on_the_cut_of_the_input() {
    const W: usize = 1000;
    for (stream, q, es_n0) in [(0u64, Qam::Qam16, 15.0), (1, Qam::Qam1024, 35.0)] {
        let mut rng = Pcg32::new(SEED_CHUNKS, stream);
        let mut x = symbols(q, 12 * W + 357, &mut rng);
        add_noise(&mut x, 10f64.powf(-es_n0 / 10.0), &mut rng);
        let x = to_c32(&x, 1.3);
        let want = estimates(&mut block(q, W), &x, W);
        assert!(want[1..].iter().all(|v| v.is_finite()));
        for plan in twenty_plans(SEED_CHUNKS) {
            let mut est = block(q, W);
            let mut out = Vec::new();
            let mut room = vec![C32::default(); 4096];
            let mut pos = 0;
            let mut sizes = plan.sizes();
            while pos < x.len() {
                let want_n = sizes.next().unwrap().min(x.len() - pos);
                let p = est.process(&x[pos..pos + want_n], &mut room).unwrap();
                out.extend_from_slice(&room[..p.produced]);
                pos += p.consumed;
                let k = est.estimates() as usize;
                let have = est.snr_db();
                let expect = if k == 0 { f64::NAN } else { want[k] };
                assert_eq!(
                    have.to_bits(),
                    expect.to_bits(),
                    "{q:?} {plan:?} at {pos}: {have} against {expect}"
                );
                assert_eq!(est.remaining(), W - pos % W);
            }
            assert_eq!(out, x, "{q:?} {plan:?}: the output is the input");
        }
    }
}

#[test]
fn the_amplitude_does_not_change_the_estimate() {
    let mut rng = Pcg32::new(SEED_CHUNKS, 2);
    let mut x = symbols(Qam::Qam64, 6 * 1024, &mut rng);
    add_noise(&mut x, 10f64.powf(-2.2), &mut rng);
    let want = estimates(&mut block(Qam::Qam64, 1024), &to_c32(&x, 1.0), 1024);
    for k in [-30i32, -20, -1, 1, 20, 30] {
        let got = estimates(
            &mut block(Qam::Qam64, 1024),
            &to_c32(&x, 2f64.powi(k)),
            1024,
        );
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got), bits(&want), "2^{k}");
    }
}

#[test]
fn extreme_inputs() {
    const W: usize = 256;
    let q = Qam::Qam16;
    // Silence: no estimate and no amplitude, also later.
    let mut est = block(q, W);
    assert!(
        estimates(&mut est, &vec![C32::default(); 4 * W], W)
            .iter()
            .all(|v| v.is_nan())
    );
    assert_eq!(est.estimates(), 0);
    // Points on the grid: a very high estimate.
    let mut rng = Pcg32::new(SEED_CHUNKS, 3);
    let points: Vec<C32> = (0..3 * W).map(|_| q.point(rng.below(16))).collect();
    let got = estimates(&mut block(q, W), &points, W);
    assert!(got[1] > 100.0 && got[2] > 100.0, "{got:?}");
    // A window with a NaN or an infinite symbol keeps the estimate and the amplitude.
    let mut noisy = symbols(q, 6 * W, &mut rng);
    add_noise(&mut noisy, 1e-2, &mut rng);
    let clean = to_c32(&noisy, 1.0);
    let want = estimates(&mut block(q, W), &clean, W);
    for bad in [f32::NAN, f32::INFINITY, -f32::INFINITY] {
        let mut x = clean.clone();
        x[3 * W + 17].re = bad;
        let mut est = block(q, W);
        let got = estimates(&mut est, &x, W);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got[..3]), bits(&want[..3]), "{bad}");
        assert_eq!(
            got[3].to_bits(),
            want[2].to_bits(),
            "{bad}: the estimate is kept"
        );
        assert_eq!(est.estimates(), 4);
        // The amplitude was kept: the windows after it are those of an input without the
        // window of the bad symbol.
        let mut skip = clean[..3 * W].to_vec();
        skip.extend_from_slice(&clean[4 * W..]);
        let mut again = estimates(&mut block(q, W), &skip, W);
        again.truncate(5);
        assert_eq!(got[4].to_bits(), again[3].to_bits(), "{bad}");
    }
    // Pure noise reads low; the floor holds.
    let mut n = vec![Complex64::new(0.0, 0.0); 8 * W];
    add_noise(&mut n, 1.0, &mut rng);
    let got = estimates(&mut block(Qam::Qam1024, W), &to_c32(&n, 1.0), W);
    assert!(
        got[1..]
            .iter()
            .all(|&v| (SNR_ESTIMATE_FLOOR_DB..3.0).contains(&v)),
        "{got:?}"
    );
    // Large and small symbols (rounded to f32 otherwise than at amplitude 1).
    for amplitude in [1e9, 1e-9] {
        let got = estimates(&mut block(q, W), &to_c32(&noisy, amplitude), W);
        for k in 1..6 {
            assert!(
                (got[k] - want[k]).abs() < 0.01,
                "{amplitude}: {got:?} {want:?}"
            );
        }
    }
}

#[test]
fn reset_parameters_and_reading() {
    let q = Qam::Qam256;
    let mut rng = Pcg32::new(SEED_CHUNKS, 4);
    let mut x = symbols(q, 3 * 512, &mut rng);
    add_noise(&mut x, 1e-3, &mut rng);
    let x = to_c32(&x, 1.0);
    let mut est = block(q, 512);
    let first = estimates(&mut est, &x, 512);
    assert_eq!(est.reading(0), Some(("snr_db", est.snr_db())));
    assert_eq!(est.reading(1), None);
    est.reset();
    assert!(est.snr_db().is_nan());
    assert_eq!(est.estimates(), 0);
    assert_eq!(est.remaining(), 512);
    let again = estimates(&mut est, &x, 512);
    assert_eq!(
        first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        again.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(est.param("window").unwrap(), ParamValue::I64(512));
    assert_eq!(est.param("constellation").unwrap(), ParamValue::Choice(5));
    assert!(est.set_param("window", ParamValue::I64(1024)).is_err());
    assert!(est.set_param("nope", ParamValue::I64(1)).is_err());
    // observe takes the symbols as process does
    let mut obs = block(q, 512);
    assert!(!obs.observe(&x[..100]));
    assert!(obs.observe(&x[100..]));
    assert_eq!(obs.snr_db().to_bits(), first[2].to_bits());
    // the configuration
    for (c, ok) in [
        (SymbolMapConstellation::Bpsk, false),
        (SymbolMapConstellation::Psk8, false),
        (SymbolMapConstellation::Qpsk, true),
        (SymbolMapConstellation::Qam1024, true),
    ] {
        let cfg = SnrEstimateConfig {
            constellation: c,
            window: 4096,
        };
        assert_eq!(cfg.validate().is_ok(), ok, "{c:?}");
    }
    for (window, ok) in [
        (63, false),
        (64, true),
        (1 << 20, true),
        ((1 << 20) + 1, false),
    ] {
        let cfg = SnrEstimateConfig {
            window,
            ..SnrEstimateConfig::default()
        };
        assert_eq!(cfg.validate().is_ok(), ok, "{window}");
        assert_eq!(
            SnrEstimateBlock::<VecStorage>::new(&cfg, rate(1), &mut VecStorage).is_ok(),
            ok
        );
    }
}

/// When the signal gives way to noise of the same power, the estimate falls to the floor region
/// within two windows (the amplitude of a poor estimate comes from the moments, not from the
/// estimate before, D2502).
#[test]
fn a_lost_signal_reads_low_at_once() {
    for (stream, q) in [(5u64, Qam::Qam64), (6, Qam::Qam256), (7, Qam::Qam1024)] {
        let mut rng = Pcg32::new(SEED_CHUNKS, stream);
        let mut x = symbols(q, 5 * WINDOW, &mut rng);
        add_noise(&mut x, 10f64.powf(-3.5), &mut rng);
        let power = x.iter().map(Complex64::norm_sqr).sum::<f64>() / x.len() as f64;
        let mut noise = vec![Complex64::new(0.0, 0.0); 6 * WINDOW];
        add_noise(&mut noise, power, &mut rng);
        x.extend_from_slice(&noise);
        let got = estimates(&mut block(q, WINDOW), &to_c32(&x, 1.0), WINDOW);
        assert!(
            got[1..5].iter().all(|v| (v - 35.0).abs() < 0.5),
            "{q:?} {got:?}"
        );
        // the first window of noise is decided at the amplitude of the signal
        assert!(got[6..].iter().all(|&v| v < 0.0), "{q:?} {got:?}");
    }
}
