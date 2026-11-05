//! Extreme inputs of the M39 blocks (test 6): `fsk_mod`, `fsk_demod` and `afc` get a stretch of
//! silence, of full-scale values and of NaN and infinities in the middle of a normal signal.
//! No call may fail or panic, and after the stretch every block works again: `fsk_mod` gives
//! exactly the deviation of the following bits, `fsk_demod` decodes every following bit after
//! the loop has locked again, `afc` returns to the offset of the signal and its output is finite.
//! The random bits come from `Pcg32` with the M39 seed 537.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod common;

use common::{rate, run_block};
use sdrk::C32;
use sdrk_blocks::{
    AfcBlock, AfcConfig, Block, FskDemodBlock, FskDemodConfig, FskModBlock, FskModConfig,
};
use sdrk_testkit::{Pcg32, VecStorage};

/// Seed of M39 test 6 (seed table of docs/milestones/README.md).
const SEED: u64 = 537;

/// The three kinds of extreme input.
#[derive(Debug, Clone, Copy)]
enum Extreme {
    /// All zeros.
    Silence,
    /// The largest finite `f32` values (with alternating signs).
    FullScale,
    /// NaN, with some positive and negative infinities among them.
    NonFinite,
}

const EXTREMES: [Extreme; 3] = [Extreme::Silence, Extreme::FullScale, Extreme::NonFinite];

/// Random soft bits `+1` / `-1`.
fn soft_bits(n: usize, stream: u64) -> Vec<f32> {
    let mut rng = Pcg32::new(SEED, stream);
    (0..n)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

/// The extreme real value number `i` of a stretch.
fn real_extreme(kind: Extreme, i: usize) -> f32 {
    match kind {
        Extreme::Silence => 0.0,
        Extreme::FullScale => {
            if i.is_multiple_of(2) {
                f32::MAX
            } else {
                -f32::MAX
            }
        }
        Extreme::NonFinite => match i % 7 {
            3 => f32::INFINITY,
            5 => f32::NEG_INFINITY,
            _ => f32::NAN,
        },
    }
}

/// Replaces `x[range]` by extreme complex values; full scale keeps the phase of the signal and
/// sets the magnitude to `f32::MAX`.
fn corrupt(x: &mut [C32], range: core::ops::Range<usize>, kind: Extreme) {
    for (i, s) in x[range].iter_mut().enumerate() {
        *s = match kind {
            Extreme::Silence => C32::new(0.0, 0.0),
            Extreme::FullScale => C32::new(s.re * f32::MAX, s.im * f32::MAX),
            Extreme::NonFinite => {
                let v = real_extreme(kind, i);
                C32::new(v, real_extreme(kind, i + 1))
            }
        };
    }
}

/// Instantaneous frequency in Hz between consecutive samples.
fn freq_hz(x: &[C32], fs: f64) -> Vec<f64> {
    x.windows(2)
        .map(|w| {
            let a = num_complex::Complex64::new(f64::from(w[0].re), f64::from(w[0].im));
            let b = num_complex::Complex64::new(f64::from(w[1].re), f64::from(w[1].im));
            let z = b * a.conj();
            z.im.atan2(z.re) * fs / std::f64::consts::TAU
        })
        .collect()
}

#[test]
fn fsk_mod_survives_extreme_bits_and_recovers() {
    let (sps, bit_rate, dev) = (8usize, 2_000u64, 1_000.0);
    let fs = (bit_rate as usize * sps) as f64;
    for kind in EXTREMES {
        // 2-FSK with the boxcar: output sample n carries the level of bit n / sps exactly.
        let cfg = FskModConfig {
            samples_per_symbol: sps,
            deviation_hz: dev,
            ..FskModConfig::default()
        };
        let mut bits = soft_bits(3000, 20);
        let clean = bits.clone();
        for (i, b) in bits[1000..1200].iter_mut().enumerate() {
            *b = real_extreme(kind, i);
        }
        let mut block = FskModBlock::new(&cfg, rate(bit_rate), &mut VecStorage).unwrap();
        let out = run_block(&mut block, &bits, None);
        assert_eq!(out.len(), bits.len() * sps, "{kind:?}");
        for (n, s) in out.iter().enumerate() {
            let m = s.re.hypot(s.im);
            assert!((m - 1.0).abs() < 1e-3, "{kind:?}: |out[{n}]| = {m}");
        }
        // From bit 1200 on every sample turns by exactly the deviation of its bit.
        let f = freq_hz(&out, fs);
        for n in 1200 * sps..out.len() {
            let want = if clean[n / sps] > 0.0 { dev } else { -dev };
            assert!(
                (f[n - 1] - want).abs() < 0.5,
                "{kind:?}: sample {n}: {} Hz",
                f[n - 1]
            );
        }
        // GFSK with 4 levels: no failure, unit magnitude everywhere.
        let cfg = FskModConfig {
            levels: 4,
            samples_per_symbol: 5,
            deviation_hz: dev,
            bt: 0.5,
            ..FskModConfig::default()
        };
        let mut block = FskModBlock::new(&cfg, rate(bit_rate), &mut VecStorage).unwrap();
        let out = run_block(&mut block, &bits, None);
        assert_eq!(out.len(), bits.len() / 2 * 5);
        assert!(out.iter().all(|s| (s.re.hypot(s.im) - 1.0).abs() < 1e-3));
    }
}

/// 2-FSK or 4-FSK from `fsk_mod` (bt 0, h = 1) at 1000 Bd and `sps` samples per symbol.
fn fsk_signal(levels: usize, sps: usize, bits: &[f32]) -> Vec<C32> {
    let bps = if levels == 4 { 2 } else { 1 };
    let cfg = FskModConfig {
        levels,
        samples_per_symbol: sps,
        deviation_hz: 500.0,
        ..FskModConfig::default()
    };
    let mut block = FskModBlock::new(&cfg, rate(1_000 * bps as u64), &mut VecStorage).unwrap();
    let mut out = vec![C32::default(); bits.len() / bps * sps];
    let p = block.process(bits, &mut out).unwrap();
    assert_eq!(p.produced, out.len());
    out
}

/// The smallest number of wrong hard bits of `got[from..]` against `want` over the shifts
/// `-64 ..= 64` (the loop may gain or lose symbols during the stretch), with the shift.
fn best_errors(want: &[f32], got: &[f32], from: usize) -> (i64, usize) {
    (-64i64..=64)
        .map(|lag| {
            let mut errors = 0usize;
            let mut compared = 0usize;
            for (i, g) in got.iter().enumerate().skip(from) {
                let j = i as i64 - lag;
                if let Some(w) = usize::try_from(j).ok().and_then(|j| want.get(j)) {
                    compared += 1;
                    errors += usize::from((*g > 0.0) != (*w > 0.0));
                }
            }
            (lag, errors + got.len().saturating_sub(from + compared))
        })
        .min_by_key(|&(_, e)| e)
        .unwrap()
}

#[test]
fn fsk_demod_survives_extreme_input_and_relocks() {
    let sps = 8usize;
    for levels in [2usize, 4] {
        let bps = if levels == 4 { 2 } else { 1 };
        let symbols = 4000usize;
        let bits = soft_bits(symbols * bps, 30 + levels as u64);
        let cfg = FskDemodConfig {
            levels,
            symbol_rate_hz: 1_000.0,
            deviation_hz: 500.0,
            ..FskDemodConfig::default()
        };
        for kind in EXTREMES {
            let mut x = fsk_signal(levels, sps, &bits);
            // 200 symbols of extreme input from symbol 1500 on.
            corrupt(&mut x, 1500 * sps..1700 * sps, kind);
            let mut block =
                FskDemodBlock::new(&cfg, rate(1_000 * sps as u64), &mut VecStorage).unwrap();
            let got = run_block(&mut block, &x, None);
            let n = got.len() / bps;
            assert!(n.abs_diff(symbols) <= 220, "{levels} {kind:?}: {n} symbols");
            // Before the stretch (after the acquisition) and from 500 symbols after it on,
            // every bit is right and every soft bit is finite.
            let (lag, errors) = best_errors(&bits[..1400 * bps], &got[..1400 * bps], 500 * bps);
            assert_eq!(errors, 0, "{levels} {kind:?}: before, lag {lag}");
            let from = (2200 * bps).min(got.len());
            assert!(
                got[from..].iter().all(|v| v.is_finite()),
                "{levels} {kind:?}: non-finite soft bits after the stretch"
            );
            let (lag, errors) = best_errors(&bits, &got, from);
            assert_eq!(errors, 0, "{levels} {kind:?}: after, lag {lag}");
            assert!(block.sps_estimate().is_finite());
            assert!((block.sps_estimate() - sps as f64).abs() < 0.1);
        }
    }
}

/// A complex tone at `hz` (continuous phase) at `fs`.
fn tone(n: usize, fs: f64, hz: f64) -> Vec<C32> {
    let mut phase = 0.0f64;
    (0..n)
        .map(|_| {
            let s = C32::new(phase.cos() as f32, phase.sin() as f32);
            phase = (phase + std::f64::consts::TAU * hz / fs) % std::f64::consts::TAU;
            s
        })
        .collect()
}

#[test]
fn afc_survives_extreme_input_and_recovers() {
    let fs = 100_000u64;
    let cfg = AfcConfig {
        max_offset_hz: 20_000.0,
        time_constant_s: 0.02,
        carrier_hz: 868e6,
        ..AfcConfig::default()
    };
    for kind in EXTREMES {
        let mut x = tone(80_000, fs as f64, 2_000.0);
        corrupt(&mut x, 20_000..25_000, kind);
        let mut afc = AfcBlock::new(&cfg, rate(fs), &mut VecStorage).unwrap();
        let out = run_block(&mut afc, &x, None);
        assert_eq!(out.len(), x.len());
        let f = afc.offset_hz();
        assert!((f - 2_000.0).abs() < 0.5, "{kind:?}: {f} Hz");
        assert!(afc.offset_ppm().is_finite());
        assert!(
            out[25_000..]
                .iter()
                .all(|s| s.re.is_finite() && s.im.is_finite()),
            "{kind:?}: non-finite output after the stretch"
        );
        // The output is back at 0 Hz.
        for v in &freq_hz(&out[70_000..], fs as f64) {
            assert!(v.abs() < 2.0, "{kind:?}: {v} Hz");
        }
        // hold during the extreme stretch keeps the estimate as it was.
        let mut afc = AfcBlock::new(&cfg, rate(fs), &mut VecStorage).unwrap();
        let mut out = vec![C32::default(); x.len()];
        afc.process(&x[..20_000], &mut out[..20_000]).unwrap();
        let before = afc.offset_hz();
        afc.set_param("hold", sdrk_blocks::ParamValue::Bool(true))
            .unwrap();
        afc.process(&x[20_000..25_000], &mut out[20_000..25_000])
            .unwrap();
        assert_eq!(afc.offset_hz(), before, "{kind:?}: hold");
    }
}
