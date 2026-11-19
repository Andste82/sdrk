//! M44.3 §1: the switch of the equalizer to the decision mode, `blind_switch` `count` or
//! `measured`. The `equalize` block with the measured switch equals the kernel whole and in 20
//! cut plans (test 6, seed 892) and switches before its CMA symbols on a converged 256-QAM
//! output; 16-QAM and 64-QAM (and the PSK constellations) give the same bits with either switch
//! (test 4: the measure never switches them before the count, D2402); `qam_demod`,
//! `qam_frame_decode` and `psk_demod` forward `blind_switch` unchanged to their equalizer
//! (rule 2 of the naming rule), and `qam_demod` with the measured switch of 256-QAM does not
//! depend on the cut plan (test 6, seed 892).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

mod common;

use common::{check_block, rate, run_block, same_bits};
use sdrk::{C32, Sdrk};
use sdrk_blocks::{
    Block, EqConstellation, EqMode, EqSwitch, EqualizeBlock, EqualizeConfig, ParamValue,
    PskDemodBlock, PskDemodConfig, QamDemodBlock, QamDemodConfig, QamFrameDecodeConfig,
    QamModBlock, QamModConfig, SymbolMapConstellation, qam_acquisition_check,
};
use sdrk_core::kernels::constellation::Qam;
use sdrk_core::kernels::equalize::{EqParams, Equalize, EqualizeState};
use sdrk_testkit::{Pcg32, VecStorage};

/// The seed of M44.3 test 6 (the chunk invariance with the criterion).
const SEED: u64 = 892;

/// Random points of `q` through the channel of D1511, `1 + (0.2 + 0.12 j) z^-1 + (-0.05 + 0.06 j)
/// z^-2` scaled by `echo` (0: no echo), turned by 0.3 rad, with Gaussian noise of `sigma` per
/// component.
fn channel_signal(q: Qam, n: usize, echo: f32, sigma: f32, stream: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED, stream);
    let s: Vec<C32> = (0..n).map(|_| q.point(rng.next_u32())).collect();
    let (sn, cs) = 0.3f32.sin_cos();
    let (h1, h2) = (
        C32::new(0.2 * echo, 0.12 * echo),
        C32::new(-0.05 * echo, 0.06 * echo),
    );
    (0..n)
        .map(|k| {
            let mut v = s[k];
            if k >= 1 {
                v += s[k - 1] * h1;
            }
            if k >= 2 {
                v += s[k - 2] * h2;
            }
            C32::new(
                v.re * cs - v.im * sn + sigma * rng.normal() as f32,
                v.re * sn + v.im * cs + sigma * rng.normal() as f32,
            )
        })
        .collect()
}

/// The configuration of `equalize` as `qam_demod` builds it (11 taps, `mu_dd` 0.005, the tracker
/// at 0.005) with the blind start of the constellation and the switch `sw`.
fn eq_config(c: EqConstellation, sw: EqSwitch) -> EqualizeConfig {
    EqualizeConfig {
        constellation: c,
        mu_dd: 0.005,
        phase_bw: 0.005,
        blind_switch: sw,
        ..EqualizeConfig::default()
    }
}

/// The symbol after which the block switched to the decision mode (runs of 64 symbols).
fn switch_symbol(block: &mut EqualizeBlock<VecStorage>, input: &[C32]) -> Option<usize> {
    let mut out = vec![C32::default(); 64];
    for (k, x) in input.chunks(64).enumerate() {
        block.process(x, &mut out[..x.len()]).unwrap();
        if block.mode() == EqMode::Decision {
            return Some(k * 64 + x.len());
        }
    }
    None
}

/// Test 6 (seed 892): the block with the measured switch of 256-QAM equals the kernel (the facade
/// whole) and does not depend on the cut plan; it switches before its 8000 CMA symbols; the
/// switch to the decision mode by the count is the same block with `blind_switch` `count`.
#[test]
fn equalize_with_the_measured_switch_equals_the_kernel_in_every_cut_plan() {
    let c = EqConstellation::Qam256;
    // Es/N0 about 42 dB; a signal on which the measure switches (it does so in about a quarter of
    // the streams here, the rule is strict, D2403).
    let input = channel_signal(Qam::Qam256, 12_000, 1.0, 0.0056, 7);
    for sw in [EqSwitch::Measured, EqSwitch::Count] {
        let cfg = eq_config(c, sw);
        assert_eq!(cfg.cma_symbols(), 8000);
        let p = EqParams {
            constellation: c,
            mu_cma: 0.02,
            mu_dd: 0.005,
            cma_symbols: 8000,
            phase_bw: 0.005,
            blind_switch: sw,
            ..EqParams::default()
        };
        let mut st =
            EqualizeState::new(vec![C32::default(); 11], vec![C32::default(); 11], p).unwrap();
        let mut want = vec![C32::default(); input.len()];
        <Sdrk as Equalize>::equalize(&Sdrk, &mut st, &input, &mut want).unwrap();
        assert_eq!(st.mode(), EqMode::Decision, "{sw:?}");
        check_block(
            &want,
            &input,
            SEED,
            || EqualizeBlock::<VecStorage>::new(&cfg, rate(25_000), &mut VecStorage).unwrap(),
            &format!("equalize {sw:?}"),
        );
        let mut block =
            EqualizeBlock::<VecStorage>::new(&cfg, rate(25_000), &mut VecStorage).unwrap();
        let at = switch_symbol(&mut block, &input).unwrap();
        let r = c.switch_rule();
        if sw == EqSwitch::Count {
            assert_eq!(at, 8000);
        } else {
            assert!(
                at >= (r.min_symbols + r.hold_symbols) as usize && at < 8000,
                "{at}"
            );
            assert_eq!(block.state().switch_hold(), r.hold_symbols);
        }
    }
}

/// Test 4: for 16-QAM, 64-QAM and the PSK constellations the measured switch gives the bits of
/// the count (the measure does not switch them before it, D2402).
#[test]
fn the_measured_switch_keeps_the_bits_of_the_count_below_256_qam() {
    for (stream, (c, q)) in [
        (EqConstellation::Qam16, Some(Qam::Qam16)),
        (EqConstellation::Qam64, Some(Qam::Qam64)),
        (EqConstellation::Qpsk, Some(Qam::Qpsk)),
        (EqConstellation::Bpsk, None),
        (EqConstellation::Psk8, None),
        (EqConstellation::Qam1024, Some(Qam::Qam1024)),
    ]
    .into_iter()
    .enumerate()
    {
        let input = channel_signal(
            q.unwrap_or(Qam::Qpsk),
            20_000,
            1.0,
            0.003,
            10 + stream as u64,
        );
        let run = |sw| {
            let mut b =
                EqualizeBlock::<VecStorage>::new(&eq_config(c, sw), rate(25_000), &mut VecStorage)
                    .unwrap();
            let out = run_block(&mut b, &input, None);
            (out, b.mode())
        };
        let (count, mc) = run(EqSwitch::Count);
        let (measured, mm) = run(EqSwitch::Measured);
        same_bits(&count, &measured, &format!("{c:?}"));
        assert_eq!((mc, mm), (EqMode::Decision, EqMode::Decision), "{c:?}");
        assert_eq!(c.switch_rule().min_symbols, u32::MAX, "{c:?}");
    }
}

/// `blind_switch` is the choice `count` (0) / `measured` (1) of `equalize`, default `measured`,
/// and `qam_demod`, `qam_frame_decode` and `psk_demod` forward it unchanged to their equalizer
/// (also without the equalizer the composite answers with its configuration).
#[test]
fn blind_switch_is_a_build_parameter_forwarded_by_the_composites() {
    let mut eq = EqualizeConfig::default();
    assert_eq!(eq.blind_switch, EqSwitch::Measured);
    assert_eq!(eq.get("blind_switch"), Ok(ParamValue::Choice(1)));
    eq.set("blind_switch", ParamValue::Choice(0)).unwrap();
    assert_eq!(eq.blind_switch, EqSwitch::Count);
    assert!(eq.set("blind_switch", ParamValue::Choice(2)).is_err());
    let block = EqualizeBlock::<VecStorage>::new(&eq, rate(1000), &mut VecStorage).unwrap();
    assert_eq!(block.param("blind_switch"), Ok(ParamValue::Choice(0)));
    assert_eq!(block.state().blind_switch(), EqSwitch::Count);
    assert_eq!(block.config(), eq);

    for taps in [11usize, 0] {
        let mut q = QamDemodConfig {
            constellation: SymbolMapConstellation::Qam256,
            symbol_rate_hz: 1000.0,
            equalizer_taps: taps,
            ..QamDemodConfig::default()
        };
        assert_eq!(q.blind_switch, EqSwitch::Measured);
        q.set("blind_switch", ParamValue::Choice(0)).unwrap();
        assert_eq!(q.get("blind_switch"), Ok(ParamValue::Choice(0)));
        if taps != 0 {
            assert_eq!(q.equalizer().unwrap().blind_switch, EqSwitch::Count);
        }
        let d = QamDemodBlock::<VecStorage>::new(&q, rate(4000), &mut VecStorage).unwrap();
        assert_eq!(d.param("blind_switch"), Ok(ParamValue::Choice(0)));
        assert_eq!(d.config(), q);
        if let Some(e) = d.equalizer() {
            assert_eq!(e.blind_switch(), EqSwitch::Count);
        }
        let mut f = QamFrameDecodeConfig::default();
        f.set("blind_switch", ParamValue::Choice(0)).unwrap();
        assert_eq!(f.demod.blind_switch, EqSwitch::Count);
        assert_eq!(f.get("blind_switch"), Ok(ParamValue::Choice(0)));
    }

    let mut p = PskDemodConfig {
        symbol_rate_hz: 1000.0,
        equalizer_taps: 11,
        ..PskDemodConfig::default()
    };
    assert_eq!(p.blind_switch, EqSwitch::Measured);
    p.set("blind_switch", ParamValue::Choice(0)).unwrap();
    assert_eq!(p.equalizer().unwrap().blind_switch, EqSwitch::Count);
    let d = PskDemodBlock::<VecStorage>::new(&p, rate(4000), &mut VecStorage).unwrap();
    assert_eq!(d.param("blind_switch"), Ok(ParamValue::Choice(0)));
    assert_eq!(d.equalizer().unwrap().blind_switch(), EqSwitch::Count);
}

/// A `qam_mod` signal of `n` random 256-QAM symbols (4 samples per symbol) turned by
/// `0.6 + 2 pi offset k / 4` at sample `k` (`offset` cycles per symbol), with Gaussian noise of
/// `sigma` per component.
fn composite_signal(n: usize, sigma: f64, offset: f64, stream: u64) -> Vec<C32> {
    let cfg = QamModConfig {
        constellation: SymbolMapConstellation::Qam256,
        samples_per_symbol: 4,
        ..QamModConfig::default()
    };
    let k = cfg.bits_per_symbol();
    let mut rng = Pcg32::new(SEED, stream);
    let bits: Vec<f32> = (0..n * k)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect();
    let mut tx = QamModBlock::<VecStorage>::new(&cfg, rate(40_000), &mut VecStorage).unwrap();
    let sig = run_block(&mut tx, &bits, None);
    let mut noise = Pcg32::new(SEED, 100 + stream);
    sig.iter()
        .enumerate()
        .map(|(i, x)| {
            let ph = 0.6 + core::f64::consts::TAU * offset * i as f64 / 4.0;
            let (s, co) = ph.sin_cos();
            let (re, im) = (f64::from(x.re), f64::from(x.im));
            C32::new(
                (re * co - im * s + sigma * noise.normal()) as f32,
                (re * s + im * co + sigma * noise.normal()) as f32,
            )
        })
        .collect()
}

/// Test 6 (seed 892): `qam_demod` with 256-QAM and the measured switch (its default) gives the
/// same output whole and in 20 cut plans, past the switch (before its 8000 CMA symbols, with the
/// hand-over over the hold) and the first check; and it switches earlier than with the count.
#[test]
fn qam_demod_with_the_measured_switch_does_not_depend_on_the_cut_plan() {
    let c = SymbolMapConstellation::Qam256;
    let n = qam_acquisition_check(c) + 2000;
    // A signal on which the measure switches (stream 6).
    let input = composite_signal(n, 0.01, 0.0, 6);
    let mut switches = Vec::new();
    for sw in [EqSwitch::Measured, EqSwitch::Count] {
        let cfg = QamDemodConfig {
            constellation: c,
            symbol_rate_hz: 10_000.0,
            equalizer_taps: 15,
            blind_switch: sw,
            ..QamDemodConfig::default()
        };
        let make =
            || QamDemodBlock::<VecStorage>::new(&cfg, rate(40_000), &mut VecStorage).unwrap();
        // The symbol of the switch, seen per call of 256 symbols.
        let mut d = make();
        let mut out = vec![0.0f32; 1 << 14];
        let mut at = None;
        for piece in input.chunks(1024) {
            let mut rest = piece;
            while !rest.is_empty() {
                let p = d.process(rest, &mut out).unwrap();
                rest = &rest[p.consumed..];
            }
            if at.is_none() && d.equalizer().unwrap().mode() == EqMode::Decision {
                at = Some(d.symbols());
            }
        }
        switches.push(at.unwrap());
        let want = run_block(&mut make(), &input, None);
        assert!(want.len() > (n - 100) * 8, "{sw:?}");
        if sw == EqSwitch::Measured {
            check_block(&want, &input, SEED, make, "qam_demod qam256 measured");
        }
    }
    assert!(switches[0] < switches[1], "{switches:?}");
}
