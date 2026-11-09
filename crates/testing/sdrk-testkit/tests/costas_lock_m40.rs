//! Test 4 of M40: the Costas loop of order 4 locks onto QPSK from any start phase (8 starts)
//! with a carrier offset of 0.01 cycles per symbol within 500 symbols at Eb/N0 10 dB, its lock
//! detector turns on, and it stays off (or turns off) on noise alone (seed 557).
//!
//! The symbols are random QPSK (Gray, `((2 b0 - 1) + j (2 b1 - 1)) / sqrt 2`, unit energy),
//! rotated by the start phase and the offset with [`FrequencyOffset`] and disturbed with
//! [`Awgn`] for Eb/N0 10 dB (one sample per symbol, two bits per symbol). The loop runs with
//! the block defaults (`loop_bw` 0.02, `damping` 0.7071, `max_offset` 0.05).
//!
//! Locked within 500 symbols means: from symbol 500 on, the outputs are the transmitted symbols
//! turned by one fixed multiple of 90 degrees (the phase ambiguity of QPSK, which the loop cannot
//! resolve) with at most 2 symbol errors and a mean phase error below 5 degrees in every window
//! of 100 symbols, the lock detector is on at symbol 500 and stays on, and the frequency
//! integrator is within 0.002 cycles per symbol of the offset.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::approx_constant
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_core::C32;
use sdrk_core::kernels::costas::{Costas, CostasConstellation, CostasState};
use sdrk_testkit::Pcg32;
use sdrk_testkit::channel::{Awgn, FrequencyOffset, Impairment};

const SEED: u64 = 557;
const SYMBOLS: usize = 3000;
const LOCK_BY: usize = 500;
const OFFSET: f64 = 0.01;
const EBN0_DB: f64 = 10.0;

fn new_loop() -> CostasState {
    CostasState::new(CostasConstellation::Qpsk, 0.02, 0.7071, 0.05).unwrap()
}

/// Random QPSK symbols of the seed.
fn qpsk(rng: &mut Pcg32, n: usize) -> Vec<Complex64> {
    let r = core::f64::consts::FRAC_1_SQRT_2;
    (0..n)
        .map(|_| {
            let b = rng.next_u32();
            Complex64::new(
                if b & 1 == 1 { r } else { -r },
                if b & 2 == 2 { r } else { -r },
            )
        })
        .collect()
}

fn to_c32(x: &[Complex64]) -> Vec<C32> {
    x.iter()
        .map(|v| C32::new(v.re as f32, v.im as f32))
        .collect()
}

#[test]
fn costas_locks_from_eight_start_phases_with_an_offset() {
    let mut rng = Pcg32::new(SEED, 40);
    let mut awgn = Awgn::from_ebn0(EBN0_DB, 2, 1.0, 1.0, SEED).unwrap();
    for start in 0..8u32 {
        // Start phases spread over the circle, off the lock points by 10 degrees.
        let phase0 = f64::from(start) * core::f64::consts::FRAC_PI_4 + 10f64.to_radians();
        let tx = qpsk(&mut rng, SYMBOLS);
        let turned: Vec<Complex64> = tx
            .iter()
            .map(|s| s * Complex64::new(phase0.cos(), phase0.sin()))
            .collect();
        let mut shifted = Vec::new();
        FrequencyOffset::new(OFFSET, 1.0)
            .unwrap()
            .process(&turned, &mut shifted);
        let mut rx = Vec::new();
        awgn.process(&shifted, &mut rx);
        let x = to_c32(&rx);
        let mut st = new_loop();
        let mut y = vec![C32::default(); SYMBOLS];
        Sdrk.costas(&mut st, &x[..LOCK_BY], &mut y[..LOCK_BY])
            .unwrap();
        assert!(
            st.locked(),
            "start {start}: not locked after {LOCK_BY} symbols"
        );
        // The rest symbol by symbol: the detector stays on.
        for k in LOCK_BY..SYMBOLS {
            Sdrk.costas(&mut st, &x[k..=k], &mut y[k..=k]).unwrap();
            assert!(st.locked(), "start {start}: lock lost at symbol {k}");
        }
        assert!(
            (st.frequency() - OFFSET).abs() < 0.002,
            "start {start}: frequency {}",
            st.frequency()
        );
        // One fixed rotation by a multiple of 90 degrees from symbol 500 on.
        let residual = |k: usize| {
            let v = Complex64::new(f64::from(y[k].re), f64::from(y[k].im)) * tx[k].conj();
            v.im.atan2(v.re)
        };
        let quarter = core::f64::consts::FRAC_PI_2;
        let turn = (residual(LOCK_BY) / quarter).round();
        let wrapped = |k: usize| {
            let d = residual(k) - turn * quarter;
            d - core::f64::consts::TAU * (d / core::f64::consts::TAU).round()
        };
        // No cycle slip: the decisions turned back by the fixed rotation are the transmitted
        // symbols (the symbol error rate at Es/N0 13 dB is about 1e-5).
        let errors = (LOCK_BY..SYMBOLS)
            .filter(|&k| wrapped(k).abs() >= quarter / 2.0)
            .count();
        assert!(
            errors <= 2,
            "start {start}: {errors} symbol errors after symbol {LOCK_BY}"
        );
        // No standing phase error: the mean over every window of 100 symbols is within 5
        // degrees (the noise alone spreads single symbols by about 9 degrees).
        for w in (LOCK_BY..SYMBOLS).step_by(100) {
            let mean = (w..w + 100).map(wrapped).sum::<f64>() / 100.0;
            assert!(
                mean.abs() < 5f64.to_radians(),
                "start {start}: mean phase error {} degrees at symbol {w}",
                mean.to_degrees()
            );
        }
    }
}

#[test]
fn the_lock_detector_stays_off_on_noise_and_turns_off_when_the_signal_ends() {
    // Noise alone at the power of the signal plus noise of the lock test: never locked.
    let zeros = vec![Complex64::new(0.0, 0.0); SYMBOLS];
    let mut noise = Vec::new();
    Awgn::new(0.7, SEED).unwrap().process(&zeros, &mut noise);
    let noise = to_c32(&noise);
    let mut st = new_loop();
    let mut y = [C32::default(); 1];
    for (k, x) in noise.iter().enumerate() {
        Sdrk.costas(&mut st, core::slice::from_ref(x), &mut y)
            .unwrap();
        assert!(!st.locked(), "locked on noise at symbol {k}");
    }
    // A locked loop turns off within 500 symbols after the signal is gone.
    let mut rng = Pcg32::new(SEED, 41);
    let tx = qpsk(&mut rng, 1000);
    let mut rx = Vec::new();
    Awgn::from_ebn0(EBN0_DB, 2, 1.0, 1.0, SEED)
        .unwrap()
        .process(&tx, &mut rx);
    let mut st = new_loop();
    let x = to_c32(&rx);
    let mut out = vec![C32::default(); x.len()];
    Sdrk.costas(&mut st, &x, &mut out).unwrap();
    assert!(st.locked());
    let mut off_at = None;
    for (k, x) in noise.iter().enumerate().take(LOCK_BY) {
        Sdrk.costas(&mut st, core::slice::from_ref(x), &mut y)
            .unwrap();
        if !st.locked() {
            off_at = Some(k);
            break;
        }
    }
    assert!(
        off_at.is_some(),
        "still locked {LOCK_BY} symbols into the noise"
    );
}
