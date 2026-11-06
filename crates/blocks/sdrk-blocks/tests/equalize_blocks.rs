//! The block of M43 §2: `EqualizeBlock` (the kernel `equalize` through the facade) equal to the
//! kernel whole and in 20 cut plans of seed 595 (test 6) for every constellation and mode, the
//! final state, the timing (ratio 1, delay `(taps - 1) / 2` symbols), the exact storage (also
//! of `psk_demod` with `equalizer_taps` and `equalizer_feedback_taps`), the runtime parameters
//! `mu_cma` and `mu_dd` (applied from the next symbol), the build parameters (`feedback_taps`,
//! `phase_bw`), `reset` and the extremes.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]

mod common;

use common::{check_block, check_exact_storage, rate, run_block, same_bits};
use sdrk::{C32, Delay, Ratio, RatioKind, Sdrk};
use sdrk_blocks::{
    ArenaStorage, Block, EqConstellation, EqMode, EqualizeBlock, EqualizeConfig, Error, ParamValue,
    PskDemodBlock, PskDemodConfig, PskVariant,
};
use sdrk_core::kernels::equalize::{EqParams, Equalize, EqualizeState};
use sdrk_testkit::{Pcg32, VecStorage};

type A<'a> = ArenaStorage<'a>;

/// Seed of the chunk plans and the random signals of M43 (test 6).
const SEED: u64 = 595;

const H: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// `n` random symbols of `c` (unit amplitude) turned by `phase`, through
/// `1 + 0.3 e^(j0.6) z^-1 + 0.1 e^(-j2) z^-2`, with complex noise of `sigma` per component.
fn channel_symbols(c: EqConstellation, n: usize, phase: f64, sigma: f64, stream: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED, stream);
    let sym: Vec<(f64, f64)> = (0..n)
        .map(|_| {
            let k = rng.next_u32();
            match c {
                EqConstellation::Bpsk => (if k & 1 == 0 { 1.0 } else { -1.0 }, 0.0),
                EqConstellation::Qpsk => (
                    if k & 1 == 0 {
                        f64::from(H)
                    } else {
                        -f64::from(H)
                    },
                    if k & 2 == 0 {
                        f64::from(H)
                    } else {
                        -f64::from(H)
                    },
                ),
                _ => {
                    let a = f64::from(2 * (k & 7) + 1) * std::f64::consts::PI / 8.0;
                    (a.cos(), a.sin())
                }
            }
        })
        .collect();
    let taps = [(1.0f64, 0.0f64), (0.3, 0.6), (0.1, -2.0)];
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (0.0, 0.0);
            for (d, &(g, p)) in taps.iter().enumerate() {
                if k >= d {
                    let (sr, si) = sym[k - d];
                    let (c, s) = ((p + phase).cos() * g, (p + phase).sin() * g);
                    re += sr * c - si * s;
                    im += sr * s + si * c;
                }
            }
            C32::new(
                (re + sigma * rng.normal()) as f32,
                (im + sigma * rng.normal()) as f32,
            )
        })
        .collect()
}

/// The kernel state of `cfg` on `Vec` storage.
fn kernel_state(cfg: &EqualizeConfig) -> EqualizeState<Vec<C32>, Vec<C32>> {
    let n = cfg.taps + cfg.feedback_taps;
    EqualizeState::new(
        vec![C32::default(); n],
        vec![C32::default(); n],
        EqParams {
            constellation: cfg.constellation,
            feedback_taps: cfg.feedback_taps,
            mu_cma: cfg.mu_cma as f32,
            mu_dd: cfg.mu_dd as f32,
            cma_symbols: cfg.cma_symbols,
            phase_bw: cfg.phase_bw,
        },
    )
    .unwrap()
}

/// The four configurations of the tests: QPSK with the defaults (CMA, then decisions, the
/// phase tracker), BPSK 9 taps with 2 feedback taps (a short CMA phase), 8PSK with 3 feedback
/// taps in decision mode from the start, QPSK 21 taps CMA only without the tracker.
fn configs() -> [EqualizeConfig; 4] {
    let d = EqualizeConfig::default();
    [
        d,
        EqualizeConfig {
            taps: 9,
            feedback_taps: 2,
            constellation: EqConstellation::Bpsk,
            mu_cma: 0.02,
            mu_dd: 0.01,
            cma_symbols: 500,
            phase_bw: 0.01,
        },
        EqualizeConfig {
            constellation: EqConstellation::Psk8,
            feedback_taps: 3,
            cma_symbols: 0,
            ..d
        },
        EqualizeConfig {
            taps: 21,
            cma_symbols: u32::MAX,
            phase_bw: 0.0,
            ..d
        },
    ]
}

#[test]
fn equalize_block_equals_the_kernel_whole_and_in_twenty_plans() {
    for (i, cfg) in configs().into_iter().enumerate() {
        // CMA is phase-blind; the decision mode needs the phase tracker, which follows a
        // constant phase within the decision region of 8PSK from the start
        let phase = if cfg.cma_symbols == u32::MAX {
            0.9
        } else {
            0.25
        };
        let input = channel_symbols(cfg.constellation, 4000, phase, 0.03, i as u64);
        let mut st = kernel_state(&cfg);
        let mut want = vec![C32::default(); input.len()];
        <Sdrk as Equalize>::equalize(&Sdrk, &mut st, &input, &mut want).unwrap();
        let what = format!("equalize case {i}");
        check_block(
            &want,
            &input,
            SEED,
            || EqualizeBlock::<VecStorage>::new(&cfg, rate(9600), &mut VecStorage).unwrap(),
            &what,
        );
        let mut block =
            EqualizeBlock::<VecStorage>::new(&cfg, rate(9600), &mut VecStorage).unwrap();
        let got = run_block(&mut block, &input, None);
        same_bits(&want, &got, &what);
        // the final state is the kernel's
        same_bits(st.weights(), block.weights(), &format!("{what} weights"));
        same_bits(
            st.history(),
            block.state().history(),
            &format!("{what} history"),
        );
        assert_eq!(block.mse().to_bits(), st.mse().to_bits(), "{what}");
        assert_eq!(block.mode(), st.mode(), "{what}");
        assert_eq!(block.state().phase(), st.phase(), "{what}");
        let mode = if cfg.cma_symbols == u32::MAX {
            EqMode::Cma
        } else {
            EqMode::Decision
        };
        assert_eq!(block.mode(), mode, "{what}");
        // the equalizer has learnt the channel (a sanity check, not a threshold of the spec)
        if mode == EqMode::Decision {
            assert!(block.mse() < 0.02, "{what}: mse {}", block.mse());
        } else {
            let tail = &got[3000..];
            let disp: f32 = tail
                .iter()
                .map(|y| (y.norm_sqr() - 1.0).powi(2))
                .sum::<f32>()
                / tail.len() as f32;
            assert!(disp < 0.05, "{what}: dispersion {disp}");
        }
        // reset: the start state, and the same output again
        block.reset();
        let fresh = kernel_state(&cfg);
        same_bits(fresh.weights(), block.weights(), &format!("{what} reset"));
        assert_eq!(block.mode(), fresh.mode());
        assert_eq!(block.mse(), 1.0);
        same_bits(&want, &run_block(&mut block, &input, None), "after reset");
    }
}

#[test]
fn equalize_timing_and_short_output() {
    for taps in [3usize, 11, 63] {
        let cfg = EqualizeConfig {
            taps,
            ..EqualizeConfig::default()
        };
        let mut b = EqualizeBlock::<VecStorage>::new(&cfg, rate(4800), &mut VecStorage).unwrap();
        let t = b.timing();
        assert_eq!(
            (t.ratio, t.kind, t.delay),
            (
                Ratio::ONE,
                RatioKind::Constant,
                Delay::new((taps as u64 - 1) / 2, 1).unwrap()
            )
        );
        assert_eq!(b.min_input(), 1);
        assert_eq!(
            EqualizeBlock::<VecStorage>::rate_out(&cfg, rate(4800)).unwrap(),
            rate(4800)
        );
        // the centre tap delays the first symbol by (taps - 1) / 2
        let x = C32::new(H, -H);
        let mut input = vec![C32::default(); taps];
        input[0] = x;
        let mut out = vec![C32::default(); taps];
        let p = b.process(&input, &mut out).unwrap();
        assert_eq!((p.consumed, p.produced), (taps, taps));
        assert_eq!(out[(taps - 1) / 2], x, "taps {taps}");
        // a short output takes as many symbols as fit
        let p = b.process(&input, &mut [C32::default(); 2]).unwrap();
        assert_eq!((p.consumed, p.produced), (2, 2));
        let p = b.process(&input, &mut []).unwrap();
        assert_eq!((p.consumed, p.produced), (0, 0));
    }
}

#[test]
fn equalize_storage_is_exact() {
    for taps in [3usize, 11, 63] {
        let cfg = EqualizeConfig {
            taps,
            ..EqualizeConfig::default()
        };
        let r = rate(9600);
        let bytes = EqualizeBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert_eq!(bytes, 2 * taps * 8, "taps {taps}");
        check_exact_storage(bytes, |a| EqualizeBlock::new(&cfg, r, a).map(|_| ()));
        let cfg = EqualizeConfig {
            feedback_taps: 5,
            ..cfg
        };
        let bytes = EqualizeBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert_eq!(bytes, 2 * (taps + 5) * 8, "taps {taps} + 5");
        check_exact_storage(bytes, |a| EqualizeBlock::new(&cfg, r, a).map(|_| ()));
    }
    // psk_demod with the equalizer: its weights and history after the other buffers
    for (order, differential, variant, taps, feedback) in [
        (2u32, false, PskVariant::Plain, 3usize, 0usize),
        (4, false, PskVariant::Plain, 11, 0),
        (4, false, PskVariant::Plain, 15, 5),
        (8, true, PskVariant::Plain, 63, 0),
        (4, true, PskVariant::Pi4, 15, 2),
        (4, false, PskVariant::Offset, 11, 3),
    ] {
        let base = PskDemodConfig {
            order,
            symbol_rate_hz: 25_000.0,
            differential,
            variant,
            ..PskDemodConfig::default()
        };
        let cfg = PskDemodConfig {
            equalizer_taps: taps,
            equalizer_feedback_taps: feedback,
            ..base
        };
        let r = rate(200_000);
        let bytes = PskDemodBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap();
        let without = PskDemodBlock::<A<'_>>::storage_bytes(&base, r).unwrap();
        assert_eq!(
            bytes,
            without + 2 * (taps + feedback) * 8,
            "{order} {variant:?} {taps}"
        );
        check_exact_storage(bytes, |a| PskDemodBlock::new(&cfg, r, a).map(|_| ()));
    }
}

#[test]
fn equalize_parameters() {
    let ok = EqualizeConfig::default();
    assert!(ok.validate().is_ok());
    for cfg in [
        EqualizeConfig { taps: 1, ..ok },
        EqualizeConfig { taps: 10, ..ok },
        EqualizeConfig { taps: 65, ..ok },
        EqualizeConfig {
            mu_cma: 0.000_009,
            ..ok
        },
        EqualizeConfig { mu_cma: 1.1, ..ok },
        EqualizeConfig {
            feedback_taps: 64,
            ..ok
        },
        EqualizeConfig {
            phase_bw: 0.000_05,
            ..ok
        },
        EqualizeConfig {
            phase_bw: 0.21,
            ..ok
        },
        EqualizeConfig {
            phase_bw: f64::NAN,
            ..ok
        },
        EqualizeConfig {
            mu_dd: f64::NAN,
            ..ok
        },
        EqualizeConfig { mu_dd: 0.0, ..ok },
    ] {
        assert!(
            EqualizeBlock::<VecStorage>::new(&cfg, rate(1000), &mut VecStorage).is_err(),
            "{cfg:?}"
        );
        assert!(EqualizeBlock::<A<'_>>::storage_bytes(&cfg, rate(1000)).is_err());
    }
    let mut b = EqualizeBlock::<VecStorage>::new(&ok, rate(1000), &mut VecStorage).unwrap();
    for (name, value) in [
        ("taps", ParamValue::I64(13)),
        ("constellation", ParamValue::Choice(0)),
        ("cma_symbols", ParamValue::I64(10)),
        ("feedback_taps", ParamValue::I64(3)),
        ("phase_bw", ParamValue::F64(0.01)),
    ] {
        assert_eq!(
            b.set_param(name, value),
            Err(Error::ReadOnlyParameter),
            "{name}"
        );
    }
    assert_eq!(b.param("taps"), Ok(ParamValue::I64(11)));
    assert_eq!(b.param("constellation"), Ok(ParamValue::Choice(1)));
    assert_eq!(b.param("cma_symbols"), Ok(ParamValue::I64(1000)));
    assert_eq!(b.param("feedback_taps"), Ok(ParamValue::I64(0)));
    assert_eq!(b.param("phase_bw"), Ok(ParamValue::F64(0.02)));
    // the build parameters through the configuration
    let mut c = ok;
    c.set("feedback_taps", ParamValue::I64(5)).unwrap();
    c.set("phase_bw", ParamValue::F64(0.0)).unwrap();
    assert_eq!((c.feedback_taps, c.phase_bw), (5, 0.0));
    assert!(matches!(
        c.set("phase_bw", ParamValue::F64(0.000_05)),
        Err(Error::InvalidParameter { .. })
    ));
    assert!(matches!(
        c.set("feedback_taps", ParamValue::I64(64)),
        Err(Error::OutOfRange { .. })
    ));
    // runtime parameters: range-checked, nothing changes on an error
    b.set_param("mu_cma", ParamValue::F64(0.004)).unwrap();
    b.set_param("mu_dd", ParamValue::F64(0.0002)).unwrap();
    assert_eq!(b.param("mu_cma"), Ok(ParamValue::F64(0.004)));
    assert_eq!(b.param("mu_dd"), Ok(ParamValue::F64(0.0002)));
    assert_eq!((b.state().mu_cma(), b.state().mu_dd()), (0.004, 0.0002));
    for v in [1.5, 0.0, f64::NAN] {
        assert!(matches!(
            b.set_param("mu_cma", ParamValue::F64(v)),
            Err(Error::OutOfRange { .. })
        ));
        assert!(matches!(
            b.set_param("mu_dd", ParamValue::F64(v)),
            Err(Error::OutOfRange { .. })
        ));
    }
    assert_eq!(
        b.set_param("mu_dd", ParamValue::I64(1)),
        Err(Error::WrongParamType)
    );
    assert_eq!((b.state().mu_cma(), b.state().mu_dd()), (0.004, 0.0002));
    assert_eq!(b.param("mu_cma"), Ok(ParamValue::F64(0.004)));
    assert_eq!(b.param("nope"), Err(Error::UnknownParameter));
    // a reset keeps the runtime values
    b.reset();
    assert_eq!(b.state().mu_cma(), 0.004);
}

/// A change of `mu_cma` or `mu_dd` applies from the next symbol: the block changed after
/// symbol `k` equals the kernel with `set_mu_*` after symbol `k`, wherever the cut is.
#[test]
fn equalize_runtime_parameters_apply_from_the_next_symbol() {
    let cfg = EqualizeConfig {
        cma_symbols: 1000,
        ..EqualizeConfig::default()
    };
    let input = channel_symbols(EqConstellation::Qpsk, 3000, 0.4, 0.03, 20);
    for (k1, k2) in [(1usize, 1001usize), (500, 2000), (999, 1000)] {
        let mut st = kernel_state(&cfg);
        let mut want = vec![C32::default(); input.len()];
        <Sdrk as Equalize>::equalize(&Sdrk, &mut st, &input[..k1], &mut want[..k1]).unwrap();
        st.set_mu_cma(0.003).unwrap();
        <Sdrk as Equalize>::equalize(&Sdrk, &mut st, &input[k1..k2], &mut want[k1..k2]).unwrap();
        st.set_mu_dd(0.002).unwrap();
        <Sdrk as Equalize>::equalize(&Sdrk, &mut st, &input[k2..], &mut want[k2..]).unwrap();

        let mut b = EqualizeBlock::<VecStorage>::new(&cfg, rate(1000), &mut VecStorage).unwrap();
        let mut got = run_block(&mut b, &input[..k1], None);
        b.set_param("mu_cma", ParamValue::F64(0.003)).unwrap();
        got.extend(run_block(&mut b, &input[k1..k2], None));
        b.set_param("mu_dd", ParamValue::F64(0.002)).unwrap();
        got.extend(run_block(&mut b, &input[k2..], None));
        same_bits(&want, &got, &format!("cut at {k1} and {k2}"));
        // a different step size changes the output, so the test sees the change
        let mut plain =
            EqualizeBlock::<VecStorage>::new(&cfg, rate(1000), &mut VecStorage).unwrap();
        assert_ne!(run_block(&mut plain, &input, None), want);
    }
}

#[test]
fn equalize_extremes_nan_infinity_silence_and_full_scale() {
    for cfg in configs() {
        // silence stays silent and leaves the start weights
        let mut b = EqualizeBlock::<VecStorage>::new(&cfg, rate(1000), &mut VecStorage).unwrap();
        let out = run_block(&mut b, &vec![C32::default(); 3000], None);
        assert!(out.iter().all(|y| *y == C32::default()));
        same_bits(kernel_state(&cfg).weights(), b.weights(), "silence weights");
        // NaN and infinity give zeros and leave the weights; the block recovers afterwards
        let mut input = channel_symbols(cfg.constellation, 3000, 0.0, 0.02, 30);
        for k in [100usize, 101, 1500, 2999] {
            input[k] = C32::new(f32::NAN, 0.5);
        }
        input[700] = C32::new(f32::INFINITY, f32::NEG_INFINITY);
        input[701] = C32::new(0.0, f32::NEG_INFINITY);
        let mut b = EqualizeBlock::<VecStorage>::new(&cfg, rate(1000), &mut VecStorage).unwrap();
        let out = run_block(&mut b, &input, None);
        for k in [100usize, 101, 700, 701, 1500, 2999] {
            assert_eq!(out[k], C32::default(), "{k}");
        }
        assert!(out.iter().all(|y| y.re.is_finite() && y.im.is_finite()));
        assert!(
            b.weights()
                .iter()
                .all(|w| w.re.is_finite() && w.im.is_finite())
        );
        assert!(b.mse().is_finite());
        // full scale (amplitude 1 on both components) stays finite
        let full: Vec<C32> = (0..3000)
            .map(|k| {
                let s = if k % 3 == 0 { -1.0 } else { 1.0 };
                C32::new(s, -s)
            })
            .collect();
        let mut b = EqualizeBlock::<VecStorage>::new(&cfg, rate(1000), &mut VecStorage).unwrap();
        let out = run_block(&mut b, &full, None);
        assert!(out.iter().all(|y| y.re.is_finite() && y.im.is_finite()));
        assert!(
            b.weights()
                .iter()
                .all(|w| w.re.is_finite() && w.im.is_finite())
        );
    }
}

#[test]
fn equalize_block_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<EqualizeBlock<VecStorage>>();
    assert_send_sync::<EqualizeBlock<A<'static>>>();
    assert_send_sync::<EqualizeConfig>();
    assert_send_sync::<EqConstellation>();
    assert_send_sync::<EqMode>();
}
