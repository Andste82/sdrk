//! Tests of M44.3 §1 in `sdrk-core`: the measured switch of `equalize` to the decision mode
//! (`EqSwitch::Measured`, the rule of `EqConstellation::switch_rule`). On 256-QAM through the
//! three-tap channel of D1511 (Eb/N0 30 dB) the kernel switches by the measure before its 8000 CMA symbols and
//! converges; the switch inside a run of the linear path and in the joint path (feedback taps)
//! does not depend on the block size; runs of `blind_left` symbols end at the switch; `reset`
//! starts the hold again; the count leaves the hold at 0; with the feature `reference`
//! `Fallback` equals `Reference` with the measured switch (also with non-finite symbols in the
//! hold). Seed 893.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::many_single_char_names
)]

use sdrk_core::kernels::constellation::Qam;
use sdrk_core::kernels::equalize::{
    EqConstellation, EqMode, EqParams, EqSwitch, Equalize, EqualizeState,
};
use sdrk_core::{C32, Fallback};

/// The seed of this file.
const SEED: u64 = 893;

/// Deterministic generator (xorshift64*), uniform in [-1, 1).
struct Rng(u64);

impl Rng {
    fn new(seed: u64, stream: u64) -> Self {
        Self(seed * 1000 + stream)
    }

    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (v as f64 / (1u64 << 52) as f64 - 1.0) as f32
    }

    fn unit_f64(&mut self) -> f64 {
        f64::from(f32::midpoint(self.next(), 1.0))
    }

    fn symbol(&mut self, q: Qam) -> u32 {
        let v = f32::midpoint(self.next(), 1.0) * q.points() as f32;
        (v as u32).min(q.points() as u32 - 1)
    }

    /// A standard normal value (Box-Muller).
    fn normal(&mut self) -> f64 {
        let u1 = self.unit_f64().max(1e-12);
        let u2 = self.unit_f64();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}

/// `n` symbols of `q` through the channel of D1511, `1 + (0.2 + 0.12 j) z^-1 + (-0.05 + 0.06 j)
/// z^-2` turned by 0.3 rad, at `ebn0_db`; the symbols and the samples.
fn d1511_signal(q: Qam, n: usize, ebn0_db: f64, stream: u64) -> (Vec<u32>, Vec<C32>) {
    let mut rng = Rng::new(SEED, stream);
    let syms: Vec<u32> = (0..n).map(|_| rng.symbol(q)).collect();
    let s: Vec<C32> = syms.iter().map(|&k| q.point(k)).collect();
    let (h1, h2) = (C32::new(0.2, 0.12), C32::new(-0.05, 0.06));
    let turn = C32::new(0.3f32.cos(), 0.3f32.sin());
    let clean: Vec<C32> = (0..n)
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
    let power = clean.iter().map(|v| f64::from(v.norm_sqr())).sum::<f64>() / n as f64;
    let n0 = 1.0 / (q.bits_per_symbol() as f64 * 10f64.powf(ebn0_db / 10.0));
    let sigma = (n0 / 2.0).sqrt() * power.sqrt();
    let x = clean
        .iter()
        .map(|v| {
            C32::new(
                v.re + (sigma * rng.normal()) as f32,
                v.im + (sigma * rng.normal()) as f32,
            )
        })
        .collect();
    (syms, x)
}

/// The parameters of the equalizer of `qam_demod` for 256-QAM (the blind start of D1517) with
/// `feedback_taps` and the switch `sw`.
fn params_256(feedback_taps: usize, sw: EqSwitch) -> EqParams {
    EqParams {
        constellation: EqConstellation::Qam256,
        feedback_taps,
        mu_cma: 0.02,
        mu_dd: 0.005,
        cma_symbols: 8000,
        phase_bw: 0.005,
        blind_switch: sw,
    }
}

fn bits(v: &[C32]) -> Vec<(u32, u32)> {
    v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

/// The equalizer in runs of `blind_left` symbols while it is blind (then in runs of 1000): each
/// switch falls on the end of a run. Returns the output and the symbol of the switch.
fn run_to_switch(st: &mut EqualizeState<Vec<C32>, Vec<C32>>, x: &[C32]) -> (Vec<C32>, usize) {
    let mut y = vec![C32::default(); x.len()];
    let mut at = 0;
    let mut switch = 0;
    while at < x.len() {
        let left = st.blind_left();
        let run = if left == 0 { 1000 } else { left as usize }.min(x.len() - at);
        let was = st.mode();
        Fallback
            .equalize(st, &x[at..at + run], &mut y[at..at + run])
            .unwrap();
        at += run;
        if was == EqMode::Cma && st.mode() == EqMode::Decision {
            switch = at;
        }
    }
    (y, switch)
}

#[test]
fn the_measured_switch_comes_before_the_count_on_256_qam_and_converges() {
    let q = Qam::Qam256;
    let n = 40_000;
    let (syms, x) = d1511_signal(q, n, 30.0, 1);
    let rule = EqConstellation::Qam256.switch_rule();
    for feedback in [0usize, 3] {
        let len = 11 + feedback;
        let mut st = EqualizeState::new(
            vec![C32::default(); len],
            vec![C32::default(); len],
            params_256(feedback, EqSwitch::Measured),
        )
        .unwrap();
        assert_eq!(st.blind_left(), rule.min_symbols + rule.hold_symbols - 1);
        let (y, switch) = run_to_switch(&mut st, &x);
        assert!(
            switch >= (rule.min_symbols + rule.hold_symbols) as usize && switch < 8000,
            "feedback {feedback}: switch at {switch}"
        );
        assert_eq!(st.switch_hold(), rule.hold_symbols);
        assert_eq!(st.blind_left(), 0);
        // The decisions of the last 10000 symbols are the symbols sent (delay 5, a quarter turn).
        let errors = (0..4u32)
            .map(|r| {
                (n - 10_000..n)
                    .filter(|&k| {
                        let mut p = q.point(syms[k - 5]);
                        for _ in 0..r {
                            p = C32::new(-p.im, p.re);
                        }
                        q.decide(y[k]) != p
                    })
                    .count()
            })
            .min()
            .unwrap();
        assert!(errors <= 20, "feedback {feedback}: {errors} errors");
        // Any block size gives the same bits (the switch inside a run of the linear path).
        let mut whole = EqualizeState::new(
            vec![C32::default(); len],
            vec![C32::default(); len],
            params_256(feedback, EqSwitch::Measured),
        )
        .unwrap();
        let mut want = vec![C32::default(); n];
        Fallback.equalize(&mut whole, &x, &mut want).unwrap();
        assert_eq!(bits(&want), bits(&y), "feedback {feedback}");
        for size in [1usize, 7, 63, 64, 65, 333, 4096] {
            let mut st2 = EqualizeState::new(
                vec![C32::default(); len],
                vec![C32::default(); len],
                params_256(feedback, EqSwitch::Measured),
            )
            .unwrap();
            let mut y2 = vec![C32::default(); n];
            for (a, b) in x.chunks(size).zip(y2.chunks_mut(size)) {
                Fallback.equalize(&mut st2, a, b).unwrap();
            }
            assert_eq!(bits(&y2), bits(&want), "feedback {feedback} size {size}");
        }
        // `reset` starts the hold and the count again.
        st.reset();
        assert_eq!((st.mode(), st.switch_hold()), (EqMode::Cma, 0));
        assert_eq!(st.blind_left(), rule.min_symbols + rule.hold_symbols - 1);
    }
}

#[test]
fn the_count_switches_at_its_symbols_and_keeps_no_hold() {
    let (_, x) = d1511_signal(Qam::Qam256, 9000, 24.0, 2);
    let mut st = EqualizeState::new(
        vec![C32::default(); 11],
        vec![C32::default(); 11],
        params_256(0, EqSwitch::Count),
    )
    .unwrap();
    assert_eq!(st.blind_left(), 8000);
    let (_, switch) = run_to_switch(&mut st, &x);
    assert_eq!((switch, st.switch_hold()), (8000, 0));
    // `u32::MAX` never switches, with either switch.
    for sw in [EqSwitch::Count, EqSwitch::Measured] {
        let p = EqParams {
            cma_symbols: u32::MAX,
            ..params_256(0, sw)
        };
        let mut st =
            EqualizeState::new(vec![C32::default(); 11], vec![C32::default(); 11], p).unwrap();
        assert_eq!(st.blind_left(), u32::MAX);
        let mut y = vec![C32::default(); x.len()];
        Fallback.equalize(&mut st, &x, &mut y).unwrap();
        assert_eq!(st.mode(), EqMode::Cma, "{sw:?}");
    }
}

#[cfg(feature = "reference")]
mod against_the_reference {
    use super::*;
    use sdrk_core::Reference;

    fn same(a: &[C32], b: &[C32]) -> bool {
        a.iter().zip(b).all(|(p, q)| {
            (p.re.to_bits() == q.re.to_bits() || (p.re.is_nan() && q.re.is_nan()))
                && (p.im.to_bits() == q.im.to_bits() || (p.im.is_nan() && q.im.is_nan()))
        })
    }

    /// `Fallback` equals `Reference` with the measured switch, linear and with feedback taps,
    /// also with a burst of non-finite symbols in the hold.
    #[test]
    fn equalize_with_the_measured_switch_fallback_equals_reference() {
        let (_, mut x) = d1511_signal(Qam::Qam256, 12_000, 24.0, 3);
        for v in &mut x[2600..2610] {
            *v = C32::new(f32::NAN, 0.0);
        }
        x[2700] = C32::new(f32::INFINITY, 1.0);
        for feedback in [0usize, 2] {
            let len = 11 + feedback;
            let p = params_256(feedback, EqSwitch::Measured);
            let mut a = EqualizeState::new(vec![C32::default(); len], vec![C32::default(); len], p)
                .unwrap();
            let mut b = a.clone();
            let mut ya = vec![C32::default(); x.len()];
            let mut yb = vec![C32::default(); x.len()];
            for (k, (xs, (oa, ob))) in x
                .chunks(97)
                .zip(ya.chunks_mut(97).zip(yb.chunks_mut(97)))
                .enumerate()
            {
                Fallback.equalize(&mut a, xs, oa).unwrap();
                Reference.equalize(&mut b, xs, ob).unwrap();
                assert!(same(oa, ob), "feedback {feedback} chunk {k}");
                assert_eq!(
                    (a.mode(), a.switch_hold(), a.blind_left()),
                    (b.mode(), b.switch_hold(), b.blind_left()),
                    "feedback {feedback} chunk {k}"
                );
            }
            assert_eq!(a.mode(), EqMode::Decision);
            assert_eq!(a.mse().to_bits(), b.mse().to_bits());
            assert!(same(a.weights(), b.weights()));
        }
    }
}
