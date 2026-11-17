//! M44.1.1 test 2 (seed 861): the building blocks `pulse_shape` and `pulse_match`.
//!
//! The taps equal `design::rrc` bit for bit (times `sqrt(sps)` for `pulse_shape`, divided by
//! `sqrt(n)` for `pulse_match`, each rounded once to `f32`); the timing, the configuration and
//! the parameters; the ranges and the rate rule of `pulse_match` with its errors (`n` within 1 %
//! of an integer from 2 to 1024); `qam_mod` and `psk_mod` give the output of `symbol_map` then
//! `pulse_shape` (then `iq_offset` for OQPSK) bit for bit, whole and in the 20 cut plans of seed
//! 861; both blocks are chunk invariant in the 20 cut plans of seed 861. The outputs against the
//! recorded baselines of M44.1 (the modulators and the matched filter of `qam_demod`) are checked
//! in `sdrk-testkit/tests/qam_baseline_m44_1.rs`.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::trivially_copy_pass_by_ref
)]

mod common;

use common::{check_block, rate, run_block, same_bits};
use sdrk::{C32, Delay, Ratio, RatioKind};
use sdrk_blocks::design::rrc;
use sdrk_blocks::{
    Block, Error, IqOffsetBlock, ParamValue, PskModBlock, PskModConfig, PskVariant,
    PulseMatchBlock, PulseMatchConfig, PulseShapeBlock, PulseShapeConfig, QamModBlock,
    QamModConfig, SymbolMapBlock, SymbolMapConfig, SymbolMapConstellation, SymbolMapVariant,
};
use sdrk_testkit::{Pcg32, VecStorage};

/// The seed of test 2.
const SEED: u64 = 861;

/// The pulses of the cases: (roll-off, span), with the ends of both ranges.
const SHAPES: [(f64, usize); 5] = [(0.35, 10), (0.2, 6), (0.5, 32), (0.05, 4), (1.0, 32)];

fn shape(sps: usize, rolloff: f64, span: usize) -> PulseShapeBlock<VecStorage> {
    PulseShapeBlock::new(
        &PulseShapeConfig {
            samples_per_symbol: sps,
            rolloff,
            span,
        },
        rate(9600),
        &mut VecStorage,
    )
    .unwrap()
}

fn matched(fs: u64, symbol_rate_hz: f64, rolloff: f64, span: usize) -> PulseMatchBlock<VecStorage> {
    PulseMatchBlock::new(
        &PulseMatchConfig {
            symbol_rate_hz,
            rolloff,
            span,
        },
        rate(fs),
        &mut VecStorage,
    )
    .unwrap()
}

/// A random complex signal of seed 861 and `stream`, components in -1 to 1.
fn complex(n: usize, stream: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED, stream);
    let mut u = || (f64::from(rng.next_u32()) / 2f64.powi(31) - 1.0) as f32;
    (0..n).map(|_| C32::new(u(), u())).collect()
}

/// Random soft bits `+-1` of seed 861 and `stream`.
fn bits(n: usize, stream: u64) -> Vec<f32> {
    let mut rng = Pcg32::new(SEED, stream);
    (0..n)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

fn design(rolloff: f64, sps: usize, span: usize) -> Vec<f64> {
    let mut h = vec![0.0f64; span * sps + 1];
    rrc(rolloff, sps, span, &mut h).unwrap();
    h
}

#[test]
fn the_taps_of_pulse_shape_are_the_design_times_sqrt_sps() {
    for sps in [2usize, 4, 5, 8, 64] {
        for (rolloff, span) in SHAPES {
            let b = shape(sps, rolloff, span);
            let gain = libm::sqrt(sps as f64);
            let want: Vec<u32> = design(rolloff, sps, span)
                .iter()
                .map(|t| ((t * gain) as f32).to_bits())
                .collect();
            let got: Vec<u32> = b.taps().iter().map(|t| t.to_bits()).collect();
            assert_eq!(got, want, "sps {sps} rolloff {rolloff} span {span}");
            assert_eq!(b.num_taps(), span * sps + 1);
            assert_eq!(b.samples_per_symbol(), sps);
            let cfg = PulseShapeConfig {
                samples_per_symbol: sps,
                rolloff,
                span,
            };
            assert_eq!(b.config(), cfg);
            // Ratio `sps`, the delay `span / 2` input symbols.
            let t = b.timing();
            assert_eq!(
                (t.ratio, t.kind),
                (Ratio::new(sps as u64, 1).unwrap(), RatioKind::Constant)
            );
            assert_eq!(
                t.delay.reduced(),
                Delay::new(span as u64, 2).unwrap().reduced()
            );
            assert_eq!(
                b.param("samples_per_symbol"),
                Ok(ParamValue::I64(sps as i64))
            );
            assert_eq!(b.param("rolloff"), Ok(ParamValue::F64(rolloff)));
            assert_eq!(b.param("span"), Ok(ParamValue::I64(span as i64)));
            assert_eq!(
                PulseShapeBlock::<VecStorage>::rate_out(&cfg, rate(9600)),
                Ok(sdrk::Rate::hz(9600 * sps as u64).unwrap())
            );
        }
    }
}

#[test]
fn the_taps_of_pulse_match_are_the_design_over_sqrt_n() {
    // (sample rate, symbol rate, n): exact multiples and rates within 1 % of one.
    for (fs, sr, n) in [
        (9600u64, 4800.0, 2usize),
        (38_400, 9600.0, 4),
        (200_000, 25_000.0, 8),
        (200_000, 24_800.0, 8),
        (48_000, 9600.0, 5),
        (1_024_000, 1000.0, 1024),
    ] {
        for (rolloff, span) in SHAPES {
            let b = matched(fs, sr, rolloff, span);
            assert_eq!(b.samples_per_symbol(), n, "{fs} {sr}");
            let gain = 1.0 / libm::sqrt(n as f64);
            let want: Vec<u32> = design(rolloff, n, span)
                .iter()
                .map(|t| ((t * gain) as f32).to_bits())
                .collect();
            let got: Vec<u32> = b.taps().iter().map(|t| t.to_bits()).collect();
            assert_eq!(got, want, "{fs} {sr} rolloff {rolloff} span {span}");
            let cfg = PulseMatchConfig {
                symbol_rate_hz: sr,
                rolloff,
                span,
            };
            assert_eq!(b.config(), cfg);
            assert_eq!(cfg.samples_per_symbol(rate(fs)), Ok(n));
            // Ratio 1, the delay `span n / 2` samples.
            let t = b.timing();
            assert_eq!((t.ratio, t.kind), (Ratio::ONE, RatioKind::Constant));
            assert_eq!(
                t.delay.reduced(),
                Delay::new((span * n) as u64, 2).unwrap().reduced()
            );
            assert_eq!(b.param("symbol_rate_hz"), Ok(ParamValue::F64(sr)));
            assert_eq!(
                PulseMatchBlock::<VecStorage>::rate_out(&cfg, rate(fs)),
                Ok(rate(fs))
            );
        }
    }
}

#[test]
fn pulse_match_keeps_the_rate_rule() {
    let rate_error = |fs: u64, sr: f64| {
        let cfg = PulseMatchConfig {
            symbol_rate_hz: sr,
            ..PulseMatchConfig::default()
        };
        let errors = [
            cfg.validate(rate(fs)).err(),
            PulseMatchBlock::new(&cfg, rate(fs), &mut VecStorage).err(),
            PulseMatchBlock::<VecStorage>::storage_bytes(&cfg, rate(fs)).err(),
            PulseMatchBlock::<VecStorage>::rate_out(&cfg, rate(fs)).err(),
        ];
        errors.iter().all(|e| {
            matches!(
                e,
                Some(Error::InvalidParameter {
                    name: "symbol_rate_hz",
                    ..
                })
            )
        })
    };
    // Within 1 % of 8: 8.07 and 7.93 samples per symbol; outside: 8.09, 7.5, 7.91.
    assert_eq!(matched(80_700, 10_000.0, 0.35, 10).samples_per_symbol(), 8);
    assert_eq!(matched(79_300, 10_000.0, 0.35, 10).samples_per_symbol(), 8);
    for (fs, sr) in [
        (80_900u64, 10_000.0),
        (75_000, 10_000.0),
        (79_100, 10_000.0),
        // below 2 samples per symbol, and above 1024
        (9600, 9600.0),
        (9600, 6400.0),
        (1_025_000, 1000.0),
        (2_000_000, 1000.0),
    ] {
        assert!(rate_error(fs, sr), "{fs} / {sr}");
    }
    // The ranges: `OutOfRange` of the parameter.
    for (cfg, name) in [
        (
            PulseMatchConfig {
                symbol_rate_hz: 0.0,
                ..PulseMatchConfig::default()
            },
            "symbol_rate_hz",
        ),
        (
            PulseMatchConfig {
                symbol_rate_hz: f64::NAN,
                ..PulseMatchConfig::default()
            },
            "symbol_rate_hz",
        ),
        (
            PulseMatchConfig {
                rolloff: 0.04,
                ..PulseMatchConfig::default()
            },
            "rolloff",
        ),
        (
            PulseMatchConfig {
                span: 33,
                ..PulseMatchConfig::default()
            },
            "span",
        ),
    ] {
        assert!(
            matches!(
                cfg.validate(rate(9600)),
                Err(Error::OutOfRange { name: n, .. }) if n == name
            ),
            "{cfg:?}"
        );
    }
    // Build parameters: read-only on the block; unknown names are unknown.
    let mut b = matched(9600, 2400.0, 0.35, 10);
    assert_eq!(
        b.set_param("span", ParamValue::I64(8)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(b.param("order"), Err(Error::UnknownParameter));
}

#[test]
fn pulse_shape_keeps_its_ranges() {
    for (cfg, name) in [
        (
            PulseShapeConfig {
                samples_per_symbol: 1,
                ..PulseShapeConfig::default()
            },
            "samples_per_symbol",
        ),
        (
            PulseShapeConfig {
                samples_per_symbol: 65,
                ..PulseShapeConfig::default()
            },
            "samples_per_symbol",
        ),
        (
            PulseShapeConfig {
                rolloff: 1.01,
                ..PulseShapeConfig::default()
            },
            "rolloff",
        ),
        (
            PulseShapeConfig {
                span: 3,
                ..PulseShapeConfig::default()
            },
            "span",
        ),
    ] {
        for e in [
            cfg.validate().err(),
            PulseShapeBlock::new(&cfg, rate(9600), &mut VecStorage).err(),
            PulseShapeBlock::<VecStorage>::storage_bytes(&cfg, rate(9600)).err(),
        ] {
            assert!(
                matches!(e, Some(Error::OutOfRange { name: n, .. }) if n == name),
                "{cfg:?}"
            );
        }
    }
    // A call fills the output room: the samples of a symbol may be split across calls (the
    // third symbol gives two of its four samples now and the rest with the next call).
    let x = complex(5, 1);
    let mut b = shape(4, 0.35, 10);
    let mut out = [C32::default(); 10];
    let p = b.process(&x, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (3, 10));
    let mut rest = [C32::default(); 10];
    let q = b.process(&x[3..], &mut rest).unwrap();
    assert_eq!((q.consumed, q.produced), (2, 10));
    let whole = run_block(&mut shape(4, 0.35, 10), &x, None);
    same_bits(&whole[..10], &out, "first call");
    same_bits(&whole[10..], &rest, "second call");
    assert_eq!(
        b.set_param("rolloff", ParamValue::F64(0.5)),
        Err(Error::ReadOnlyParameter)
    );
}

/// The output of `symbol_map` then `pulse_shape` (then `iq_offset`), each block run whole.
fn building_blocks(
    map: &SymbolMapConfig,
    pulse: &PulseShapeConfig,
    offset: Option<sdrk_blocks::IqOffsetConfig>,
    input: &[f32],
) -> Vec<C32> {
    let r = rate(48_000);
    let symbols = run_block(
        &mut SymbolMapBlock::<VecStorage>::new(map, r, &mut VecStorage).unwrap(),
        input,
        None,
    );
    let y = run_block(
        &mut PulseShapeBlock::<VecStorage>::new(pulse, r, &mut VecStorage).unwrap(),
        &symbols,
        None,
    );
    match offset {
        Some(c) => run_block(
            &mut IqOffsetBlock::<VecStorage>::new(&c, r, &mut VecStorage).unwrap(),
            &y,
            None,
        ),
        None => y,
    }
}

#[test]
fn the_modulators_are_symbol_map_then_pulse_shape() {
    let r = rate(48_000);
    let mut stream = 10u64;
    for c in [SymbolMapConstellation::Qam16, SymbolMapConstellation::Qam64] {
        for (sps, (rolloff, span)) in [(8usize, SHAPES[0]), (5, SHAPES[1]), (2, SHAPES[2])] {
            let cfg = QamModConfig {
                constellation: c,
                samples_per_symbol: sps,
                rolloff,
                span,
            };
            let input = bits(600 * cfg.bits_per_symbol(), stream);
            stream += 1;
            let map = SymbolMapConfig {
                constellation: c,
                ..SymbolMapConfig::default()
            };
            let want = building_blocks(&map, &cfg.pulse(), None, &input);
            assert_eq!(want.len(), 600 * sps);
            check_block(
                &want,
                &input,
                SEED,
                || QamModBlock::<VecStorage>::new(&cfg, r, &mut VecStorage).unwrap(),
                &format!("{cfg:?}"),
            );
        }
    }
    for (c, differential, variant) in [
        (SymbolMapConstellation::Bpsk, false, PskVariant::Plain),
        (SymbolMapConstellation::Qpsk, true, PskVariant::Pi4),
        (SymbolMapConstellation::Psk8, true, PskVariant::Plain),
        (SymbolMapConstellation::Qpsk, false, PskVariant::Offset),
    ] {
        let cfg = PskModConfig {
            constellation: c,
            samples_per_symbol: 6,
            differential,
            variant,
            ..PskModConfig::default()
        };
        let input = bits(600 * cfg.bits_per_symbol(), stream);
        stream += 1;
        let map = SymbolMapConfig {
            constellation: c,
            differential,
            variant: if variant == PskVariant::Pi4 {
                SymbolMapVariant::Pi4
            } else {
                SymbolMapVariant::Plain
            },
        };
        assert_eq!(cfg.offset().is_some(), variant == PskVariant::Offset);
        let want = building_blocks(&map, &cfg.pulse(), cfg.offset(), &input);
        check_block(
            &want,
            &input,
            SEED,
            || PskModBlock::<VecStorage>::new(&cfg, r, &mut VecStorage).unwrap(),
            &format!("{cfg:?}"),
        );
    }
}

#[test]
fn the_pulse_blocks_are_chunk_invariant() {
    for (stream, (sps, (rolloff, span))) in [(8usize, SHAPES[0]), (5, SHAPES[1]), (64, SHAPES[2])]
        .into_iter()
        .enumerate()
    {
        let x = complex(3000, 100 + stream as u64);
        let want = run_block(&mut shape(sps, rolloff, span), &x, None);
        assert_eq!(want.len(), 3000 * sps);
        check_block(
            &want,
            &x,
            SEED,
            || shape(sps, rolloff, span),
            &format!("pulse_shape {sps}"),
        );
        // `reset` starts again from zeros.
        let mut b = shape(sps, rolloff, span);
        let first = run_block(&mut b, &x, None);
        b.reset();
        same_bits(&first, &run_block(&mut b, &x, None), "pulse_shape reset");
    }
    for (stream, (fs, sr, (rolloff, span))) in [
        (200_000u64, 25_000.0, SHAPES[0]),
        (48_000, 9600.0, SHAPES[1]),
        (9600, 4800.0, SHAPES[2]),
    ]
    .into_iter()
    .enumerate()
    {
        let x = complex(20_000, 200 + stream as u64);
        let want = run_block(&mut matched(fs, sr, rolloff, span), &x, None);
        assert_eq!(want.len(), x.len());
        check_block(
            &want,
            &x,
            SEED,
            || matched(fs, sr, rolloff, span),
            &format!("pulse_match {fs} {sr}"),
        );
        let mut b = matched(fs, sr, rolloff, span);
        let first = run_block(&mut b, &x, None);
        b.reset();
        same_bits(&first, &run_block(&mut b, &x, None), "pulse_match reset");
    }
}
