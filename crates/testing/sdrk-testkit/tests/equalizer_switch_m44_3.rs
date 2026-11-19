//! M44.3 test 1 (seed 890): the switch of the equalizer to the decision mode by the measure
//! (`blind_switch` `measured`) against the counts of D1517, as a regression guard of the table of
//! D2402 and D2403.
//!
//! - **The signals of D1511** (the kernel with the settings of `qam_demod`: 11 taps, `mu_dd`
//!   0.005, the tracker at 0.005, the blind start of D1517): random symbols of every square order
//!   through the symbol-spaced channel `1 + (0.2 + 0.12 j) z^-1 + (-0.05 + 0.06 j) z^-2` turned
//!   by 0.3 rad, 8 seeds of 80000 symbols at both Eb/N0 of the BER tests (4-QAM 8 and 10 dB,
//!   16-QAM 12 and 14, 64-QAM 18 and 20, 256-QAM 22 and 24, 1024-QAM 27 and 29). Converged: the
//!   symbol error rate of the last 10000 symbols (the best delay and quarter turn) below
//!   `2 SER_awgn + 0.001` (D1511); a false switch: switched and not converged; the acquisition:
//!   the first multiple of 1000 symbols from which the symbol error rate of the rest stays below
//!   that bound and no window of 2000 symbols is above `max(5 bound, 0.01)`; the MER after the
//!   switch: data-aided over the 10000 symbols after it.
//! - **`Multipath::mild`** (the composite of D1517: `qam_mod` 256-QAM at 4 samples per symbol,
//!   `Multipath::mild`, AWGN, `qam_demod` with 15 taps, 8 receivers at each Eb/N0 of 256-QAM and
//!   one over AWGN, 60000 symbols): the symbol of the switch and of the lock.
//!
//! Every order except 256-QAM keeps the bits of the count (its rule never switches before it);
//! 256-QAM switches by the measure no later than the count (earlier in some runs), converges in
//! every run without a false switch, acquires no later than with the count and keeps the median
//! and the minimum MER after the switch; through `mild` every receiver locks no later than with
//! the count, over AWGN earlier (D2403, D2404).

#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::print_stderr,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::large_stack_frames
)]

use num_complex::Complex64;
use sdrk::Rate;
use sdrk_blocks::{
    Block, QamDemodBlock, QamDemodConfig, QamModBlock, QamModConfig, SymbolMapConstellation,
};
use sdrk_core::kernels::constellation::Qam;
use sdrk_core::kernels::equalize::{
    EqConstellation, EqMode, EqParams, EqSwitch, Equalize, EqualizeState,
};
use sdrk_core::{C32, Fallback};
use sdrk_testkit::channel::{Awgn, Impairment, Multipath};
use sdrk_testkit::{Pcg32, VecStorage};

/// The seed of M44.3 test 1.
const SEED: u64 = 890;
/// Seeds (noise and symbols) per order and Eb/N0.
const SEEDS: u64 = 8;
/// Symbols per run (D1511).
const N: usize = 80_000;

/// A standard normal value from two uniform values of the generator (Box-Muller).
fn normal(rng: &mut Pcg32) -> f64 {
    let u1 = (f64::from(rng.next_u32()) + 0.5) / 4_294_967_296.0;
    let u2 = (f64::from(rng.next_u32()) + 0.5) / 4_294_967_296.0;
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

fn sigma_of(q: Qam, ebn0_db: f64) -> f64 {
    let n0 = 1.0 / (q.bits_per_symbol() as f64 * 10f64.powf(ebn0_db / 10.0));
    (n0 / 2.0).sqrt()
}

fn eq_of(q: Qam) -> EqConstellation {
    match q {
        Qam::Qpsk => EqConstellation::Qpsk,
        Qam::Qam16 => EqConstellation::Qam16,
        Qam::Qam64 => EqConstellation::Qam64,
        Qam::Qam256 => EqConstellation::Qam256,
        _ => EqConstellation::Qam1024,
    }
}

/// The blind start of D1517 (`EqualizeConfig::blind_start`).
fn blind_start(q: Qam) -> (f32, u32) {
    match q {
        Qam::Qam256 => (0.02, 8000),
        Qam::Qam1024 => (0.005, 16_000),
        _ => (0.05, 1000),
    }
}

/// The Eb/N0 of the BER tests per order.
fn ber_snrs(q: Qam) -> [f64; 2] {
    match q {
        Qam::Qpsk => [8.0, 10.0],
        Qam::Qam16 => [12.0, 14.0],
        Qam::Qam64 => [18.0, 20.0],
        Qam::Qam256 => [22.0, 24.0],
        _ => [27.0, 29.0],
    }
}

fn quarter(p: C32, r: u32) -> C32 {
    (0..r).fold(p, |p, _| C32::new(-p.im, p.re))
}

/// The symbols, the received samples and the convergence bound of one run on the channel of
/// D1511.
fn d1511_signal(q: Qam, ebn0: f64, seed_index: u64) -> (Vec<C32>, Vec<C32>, f64) {
    let stream = 1000 * q.bits_per_axis() as u64 + 10 * seed_index;
    let mut rng = Pcg32::new(SEED, stream);
    let s: Vec<C32> = (0..N)
        .map(|_| q.point(rng.below(q.points() as u32)))
        .collect();
    let (h1, h2) = (C32::new(0.2, 0.12), C32::new(-0.05, 0.06));
    let turn = C32::new(0.3f32.cos(), 0.3f32.sin());
    let clean: Vec<C32> = (0..N)
        .map(|k| {
            let mut v = s[k];
            if k >= 1 {
                v += s[k - 1] * h1;
            }
            if k >= 2 {
                v += s[k - 2] * h2;
            }
            v * turn
        })
        .collect();
    let power = clean.iter().map(|v| f64::from(v.norm_sqr())).sum::<f64>() / N as f64;
    let sigma = sigma_of(q, ebn0) * power.sqrt();
    let mut g = Pcg32::new(SEED, stream + 500_000);
    let x = clean
        .iter()
        .map(|v| {
            C32::new(
                v.re + (sigma * normal(&mut g)) as f32,
                v.im + (sigma * normal(&mut g)) as f32,
            )
        })
        .collect();
    // The symbol error rate of the decision over AWGN at the same Eb/N0.
    let sig1 = sigma_of(q, ebn0);
    let errors = s
        .iter()
        .filter(|p| {
            let v = C32::new(
                p.re + (sig1 * normal(&mut g)) as f32,
                p.im + (sig1 * normal(&mut g)) as f32,
            );
            q.decide(v) != **p
        })
        .count();
    (s, x, 2.0 * (errors as f64 / N as f64) + 0.001)
}

/// One run of the kernel: the output and the symbol of the switch (seen per run of 64).
fn run_kernel(q: Qam, x: &[C32], sw: EqSwitch) -> (Vec<C32>, usize) {
    let (mu_cma, cma_symbols) = blind_start(q);
    let p = EqParams {
        constellation: eq_of(q),
        mu_cma,
        mu_dd: 0.005,
        cma_symbols,
        phase_bw: 0.005,
        blind_switch: sw,
        ..EqParams::default()
    };
    let mut st = EqualizeState::new(vec![C32::default(); 11], vec![C32::default(); 11], p).unwrap();
    let mut y = vec![C32::default(); x.len()];
    let mut at = usize::MAX;
    for (k, (a, b)) in x.chunks(64).zip(y.chunks_mut(64)).enumerate() {
        Fallback.equalize(&mut st, a, b).unwrap();
        if at == usize::MAX && st.mode() == EqMode::Decision {
            at = (k + 1) * 64;
        }
    }
    (y, at)
}

/// The outcome of a run: converged, the acquisition and the MER after the switch.
struct Outcome {
    converged: bool,
    acquisition: Option<usize>,
    mer_after: f64,
}

fn outcome(q: Qam, s: &[C32], y: &[C32], bound: f64, switch: usize) -> Outcome {
    let tail = 10_000;
    let mut best = (0usize, 0u32, usize::MAX);
    for d in 0..32usize {
        for r in 0..4 {
            let e = (N - tail..N)
                .filter(|&k| q.decide(y[k]) != quarter(s[k - d], r))
                .count();
            if e < best.2 {
                best = (d, r, e);
            }
        }
    }
    let (d, r, e_end) = best;
    let mut cum = vec![0u64; N + 1];
    for k in 0..N {
        let e = k >= d && q.decide(y[k]) != quarter(s[k - d], r);
        cum[k + 1] = cum[k] + u64::from(e);
    }
    let ser = |a: usize, b: usize| (cum[b] - cum[a]) as f64 / (b - a) as f64;
    let acquisition = (0..N - 2000).step_by(1000).find(|&t| {
        ser(t, N) < bound
            && (t..N - 1999)
                .step_by(2000)
                .all(|w| ser(w, w + 2000) < (5.0 * bound).max(0.01))
    });
    let (mut es, mut ee) = (0.0f64, 0.0f64);
    for k in switch.max(d)..(switch + 10_000).min(N) {
        let p = quarter(s[k - d], r);
        es += f64::from(p.norm_sqr());
        ee += f64::from((y[k] - p).norm_sqr());
    }
    Outcome {
        converged: (e_end as f64 / tail as f64) < bound,
        acquisition,
        mer_after: 10.0 * (es / ee).log10(),
    }
}

/// One seed of test 1: the switch and the outcome by the count, the same by the measure, and
/// whether both outputs have the same bits.
type Run = (usize, Outcome, usize, Outcome, bool);

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Test 1 on the signals of D1511: the measured switch against the count for every order.
#[test]
fn the_measured_switch_against_the_count_on_the_signals_of_d1511() {
    for q in Qam::ALL {
        for ebn0 in ber_snrs(q) {
            let runs: Vec<Run> = std::thread::scope(|sc| {
                let h: Vec<_> = (0..SEEDS)
                    .map(|i| {
                        sc.spawn(move || {
                            let (s, x, bound) = d1511_signal(q, ebn0, i);
                            let (yc, swc) = run_kernel(q, &x, EqSwitch::Count);
                            let (ym, swm) = run_kernel(q, &x, EqSwitch::Measured);
                            let same = yc.iter().zip(&ym).all(|(a, b)| {
                                a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()
                            });
                            let oc = outcome(q, &s, &yc, bound, swc);
                            let om = outcome(q, &s, &ym, bound, swm);
                            (swc, oc, swm, om, same)
                        })
                    })
                    .collect();
                h.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let show =
                |f: &dyn Fn(&Run) -> String| runs.iter().map(f).collect::<Vec<_>>().join(" ");
            let acq = |a: Option<usize>| a.map_or("-".to_string(), |a| a.to_string());
            eprintln!(
                "{q:?} {ebn0} dB: switch count [{}] measured [{}]; acquisition count [{}] measured [{}]; MER after count [{}] measured [{}]",
                show(&|r| r.0.to_string()),
                show(&|r| r.2.to_string()),
                show(&|r| acq(r.1.acquisition)),
                show(&|r| acq(r.3.acquisition)),
                show(&|r| format!("{:.1}", r.1.mer_after)),
                show(&|r| format!("{:.1}", r.3.mer_after)),
            );
            for (i, r) in runs.iter().enumerate() {
                assert!(r.1.converged && r.3.converged, "{q:?} {ebn0} dB seed {i}");
                assert!(r.1.acquisition.is_some(), "{q:?} {ebn0} dB seed {i}");
            }
            if q == Qam::Qam256 {
                let mut after_c = Vec::new();
                let mut after_m = Vec::new();
                assert!(
                    runs.iter().any(|r| r.2 < r.0),
                    "{q:?} {ebn0} dB: no earlier switch"
                );
                for (i, r) in runs.iter().enumerate() {
                    assert!(
                        r.2 <= r.0,
                        "{q:?} {ebn0} dB seed {i}: switch {} against {}",
                        r.2,
                        r.0
                    );
                    let (ac, am) = (r.1.acquisition.unwrap(), r.3.acquisition.unwrap());
                    assert!(
                        am <= ac,
                        "{q:?} {ebn0} dB seed {i}: acquisition {am} against {ac}"
                    );
                    after_c.push(r.1.mer_after);
                    after_m.push(r.3.mer_after);
                }
                let (mc, mm) = (median(after_c.clone()), median(after_m.clone()));
                let (lc, lm) = (
                    after_c.iter().copied().fold(f64::INFINITY, f64::min),
                    after_m.iter().copied().fold(f64::INFINITY, f64::min),
                );
                assert!(
                    mm >= mc && lm >= lc,
                    "{q:?} {ebn0} dB: MER after {mm} / {lm} against {mc} / {lc}"
                );
            } else {
                for (i, r) in runs.iter().enumerate() {
                    assert!(r.4, "{q:?} {ebn0} dB seed {i}: not the bits of the count");
                }
            }
        }
    }
}

/// Samples per symbol of the composite runs.
const SPS: usize = 4;
/// The symbol rate of the composite runs.
const SYMBOL_RATE: u64 = 10_000;
/// Symbols of a composite run of 256-QAM.
const N_COMPOSITE: usize = 60_000;
/// Receivers per Eb/N0 through `Multipath::mild`.
const RECEIVERS: u64 = 8;

/// One receiver of the composite: the symbol of the switch (seen per call of 1024 symbols), the
/// symbol from which the carrier loop reports lock to the end, and the symbol error rate of the
/// last 10000 symbols at the best lag and quarter turn.
struct Receiver {
    switch: Option<u64>,
    lock: Option<u64>,
    ser_end: f64,
}

fn symbols_of(soft: &[f32], k: usize) -> Vec<u32> {
    soft.chunks_exact(k)
        .map(|b| b.iter().fold(0u32, |g, &s| (g << 1) | u32::from(s > 0.0)))
        .collect()
}

/// `qam_mod` (256-QAM, 4 samples per symbol) -> `Multipath::mild` (or none) -> AWGN at `ebn0`
/// -> `qam_demod` with 15 taps and the switch `sw`.
fn composite_run(mild: bool, ebn0: f64, seed_index: u64, sw: EqSwitch) -> Receiver {
    let q = Qam::Qam256;
    let b = q.bits_per_symbol();
    let stream = 100 * q.bits_per_axis() as u64 + 10 * seed_index + u64::from(mild);
    let mut rng = Pcg32::new(SEED, 7000 + stream);
    let soft_tx: Vec<f32> = (0..(N_COMPOSITE + 64) * b)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect();
    let sent = symbols_of(&soft_tx, b);
    let mcfg = QamModConfig {
        constellation: SymbolMapConstellation::Qam256,
        samples_per_symbol: SPS,
        ..QamModConfig::default()
    };
    let rate = Rate::hz(SYMBOL_RATE * b as u64).unwrap();
    let mut tx = QamModBlock::<VecStorage>::new(&mcfg, rate, &mut VecStorage).unwrap();
    let mut sig = vec![C32::default(); N_COMPOSITE * SPS];
    let p = tx.process(&soft_tx, &mut sig).unwrap();
    sig.truncate(p.produced);
    let wide: Vec<Complex64> = sig
        .iter()
        .map(|c| Complex64::new(f64::from(c.re), f64::from(c.im)))
        .collect();
    let faded = if mild {
        let mut m = Multipath::mild(SPS as f64).unwrap();
        let mut out = Vec::new();
        m.process(&wide, &mut out);
        out
    } else {
        wide
    };
    let body = &faded[faded.len() / 4..];
    let power = body.iter().map(Complex64::norm_sqr).sum::<f64>() / body.len() as f64;
    let mut awgn =
        Awgn::from_ebn0(ebn0, b as u32, SPS as f64, power, SEED * 1000 + stream).unwrap();
    let mut noisy = Vec::new();
    awgn.process(&faded, &mut noisy);
    let input: Vec<C32> = noisy
        .iter()
        .map(|c| C32::new(c.re as f32, c.im as f32))
        .collect();
    let cfg = QamDemodConfig {
        constellation: SymbolMapConstellation::Qam256,
        symbol_rate_hz: SYMBOL_RATE as f64,
        equalizer_taps: 15,
        blind_switch: sw,
        ..QamDemodConfig::default()
    };
    let mut d = QamDemodBlock::new(
        &cfg,
        Rate::hz(SYMBOL_RATE * SPS as u64).unwrap(),
        &mut VecStorage,
    )
    .unwrap();
    let mut out = vec![0.0f32; 1 << 16];
    let mut soft = Vec::new();
    let mut switch = None;
    let mut lock = None;
    for piece in input.chunks(1024 * SPS) {
        let mut rest = piece;
        while !rest.is_empty() {
            let p = d.process(rest, &mut out).unwrap();
            soft.extend_from_slice(&out[..p.produced]);
            rest = &rest[p.consumed..];
        }
        if switch.is_none() && d.equalizer().unwrap().mode() == EqMode::Decision {
            switch = Some(d.symbols());
        }
        if d.locked() {
            lock.get_or_insert(d.symbols());
        } else {
            lock = None;
        }
    }
    let rx = symbols_of(&soft, b);
    let len = rx.len();
    let tail = 10_000;
    let mut best = usize::MAX;
    for r in 0..4u32 {
        for lag in 0..200usize {
            let e = (len - tail..len)
                .filter(|&j| {
                    j >= lag
                        && sdrk_core::kernels::constellation::rotate_symbol(rx[j], 4, (4 - r) % 4)
                            != sent[j - lag]
                })
                .count();
            best = best.min(e);
        }
    }
    Receiver {
        switch,
        lock,
        ser_end: best as f64 / tail as f64,
    }
}

/// Test 1 through `Multipath::mild` (D2404): `qam_demod` with 256-QAM and the measured switch
/// locks every receiver no later than with the count, at the same symbol error rate, and over
/// AWGN earlier.
#[test]
fn the_measured_switch_of_256_qam_through_mild_in_qam_demod() {
    for (mild, ebn0, receivers) in [
        (true, 22.0, RECEIVERS),
        (true, 24.0, RECEIVERS),
        (false, 22.0, 1),
        (false, 24.0, 1),
    ] {
        let runs: Vec<(Receiver, Receiver)> = std::thread::scope(|sc| {
            let h: Vec<_> = (0..receivers)
                .map(|i| {
                    sc.spawn(move || {
                        (
                            composite_run(mild, ebn0, i, EqSwitch::Count),
                            composite_run(mild, ebn0, i, EqSwitch::Measured),
                        )
                    })
                })
                .collect();
            h.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (i, (c, m)) in runs.iter().enumerate() {
            eprintln!(
                "mild {mild} {ebn0} dB rx {i}: switch {:?} / {:?}, lock {:?} / {:?}, SER {:.5} / {:.5}",
                c.switch, m.switch, c.lock, m.lock, c.ser_end, m.ser_end
            );
            let (lc, lm) = (c.lock.unwrap(), m.lock.unwrap());
            assert!(
                lm <= lc,
                "mild {mild} {ebn0} dB rx {i}: lock {lm} against {lc}"
            );
            assert!(
                m.switch.unwrap() <= c.switch.unwrap(),
                "mild {mild} {ebn0} dB rx {i}"
            );
            assert!(
                m.ser_end <= c.ser_end + 1e-4,
                "mild {mild} {ebn0} dB rx {i}"
            );
            if !mild {
                assert!(lm < lc, "{ebn0} dB: lock {lm} against {lc}");
            }
        }
    }
}
