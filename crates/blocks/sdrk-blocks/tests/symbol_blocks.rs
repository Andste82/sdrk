//! The blocks of M38: `SymbolSyncBlock` (the kernel `symbol_sync` through the facade: whole and
//! in 20 cut plans of seed 493, exact ratio from the symbol rate, the rate rule, no storage,
//! output-full resubmission) and `SliceBlock` (the table of §3 at corner values, the QPSK bit
//! order, the runtime `scale`, the timing; test 4 of M38).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod common;

use common::{check_block, check_exact_storage, rate, run_block};
use num_complex::Complex64;
use sdrk::{C32, CQ15, Delay, Ratio, RatioKind, Sample, Sdrk};
use sdrk_blocks::{
    ArenaStorage, Block, Error, ParamValue, SliceBlock, SliceConfig, SliceConstellation,
    StreamElem, SymbolSyncBlock, SymbolSyncConfig, Ted,
};
use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncState};
use sdrk_device::Rate;
use sdrk_testkit::r#gen::{Psk, psk_test};
use sdrk_testkit::{Pcg32, VecStorage};

/// Seed of the cut plans of M38 (seed table of docs/milestones/README.md, test 1).
const SEED: u64 = 493;

type A<'a> = ArenaStorage<'a>;

/// A shaped PSK signal with noise: `n` samples at `sps` samples per symbol, symbol 0 at
/// `offset` samples, Gaussian noise of standard deviation `sigma` per component.
fn psk_signal(
    psk: Psk,
    sps: f64,
    offset: f64,
    n: usize,
    sigma: f64,
    stream: u64,
) -> Vec<Complex64> {
    let mut rng = Pcg32::new(SEED, stream);
    let symbols = (n as f64 / sps) as usize + 40;
    let bits: Vec<u8> = (0..symbols * psk.bits_per_symbol())
        .map(|_| (rng.next_u32() & 1) as u8)
        .collect();
    let mut shaped = vec![Complex64::new(0.0, 0.0); n + 64];
    psk_test(psk, sps, 0.35, &bits, &mut shaped);
    let skip = offset.round() as usize;
    shaped[skip..skip + n]
        .iter()
        .map(|c| c + Complex64::new(sigma * rng.normal(), sigma * rng.normal()))
        .collect()
}

fn c32_input(sps: f64, ted: Ted, n: usize, stream: u64) -> Vec<C32> {
    let psk = if ted == Ted::MuellerMuller {
        Psk::Qpsk
    } else {
        Psk::Bpsk
    };
    psk_signal(psk, sps, 7.0, n, 0.05, stream)
        .iter()
        .map(|c| C32::new(c.re as f32, c.im as f32))
        .collect()
}

fn f32_input(sps: f64, n: usize, stream: u64) -> Vec<f32> {
    psk_signal(Psk::Bpsk, sps, 5.0, n, 0.05, stream)
        .iter()
        .map(|c| c.re as f32)
        .collect()
}

/// The kernel through the facade with the documented `sps`, in one call.
fn kernel_run<T: Sample + StreamElem>(cfg: &SymbolSyncConfig, fs: Rate, input: &[T]) -> Vec<T>
where
    Sdrk: SymbolSync<T>,
{
    let sps = cfg.sps(fs).unwrap();
    let mut st = SymbolSyncState::<T>::new(
        sps,
        cfg.ted,
        cfg.loop_bw,
        cfg.damping,
        cfg.max_deviation,
        cfg.ted_gain,
    )
    .unwrap();
    let mut out = vec![T::default(); input.len() / 2 + 8];
    let p = <Sdrk as SymbolSync<T>>::symbol_sync(&Sdrk, &mut st, input, &mut out).unwrap();
    assert_eq!(p.consumed, input.len());
    out.truncate(p.produced);
    out
}

/// The configurations of the block tests: (symbol rate, sample rate, sps, detector).
const CASES: [(f64, u64, f64, Ted); 5] = [
    (1000.0, 4100, 4.1, Ted::Gardner),
    (2375.0, 19_000, 8.0, Ted::Gardner),
    (1187.5, 12_000, 12_000.0 / 1187.5, Ted::Gardner),
    (24_000.0, 48_000, 2.0, Ted::MuellerMuller),
    (1200.0, 4_920, 4.1, Ted::MuellerMuller),
];

#[test]
fn symbol_sync_block_equals_the_kernel_whole_and_in_cut_plans() {
    for (i, &(sym, fs, sps, ted)) in CASES.iter().enumerate() {
        let cfg = SymbolSyncConfig {
            symbol_rate_hz: sym,
            ted,
            loop_bw: 0.02,
            ..SymbolSyncConfig::default()
        };
        let r = rate(fs);
        assert_eq!(cfg.sps(r).unwrap(), sps, "case {i}");
        let input = c32_input(sps, ted, 6000, i as u64);
        let want = kernel_run::<C32>(&cfg, r, &input);
        assert!(want.len() > 6000 / (sps as usize + 1), "case {i}");
        check_block(
            &want,
            &input,
            SEED,
            || SymbolSyncBlock::<C32, VecStorage>::new(&cfg, r, &mut VecStorage).unwrap(),
            &format!("symbol_sync c32 case {i}"),
        );
        let input = f32_input(sps, 6000, 10 + i as u64);
        let want = kernel_run::<f32>(&cfg, r, &input);
        check_block(
            &want,
            &input,
            SEED,
            || SymbolSyncBlock::<f32, VecStorage>::new(&cfg, r, &mut VecStorage).unwrap(),
            &format!("symbol_sync f32 case {i}"),
        );
    }
}

/// With a tiny output buffer the block stops when it is full; resubmitting the rest gives the
/// same samples as one call, and `reset` returns to the state after construction.
#[test]
fn symbol_sync_stops_on_a_full_output_and_resets() {
    let cfg = SymbolSyncConfig {
        symbol_rate_hz: 1000.0,
        ..SymbolSyncConfig::default()
    };
    let r = rate(4100);
    let input = c32_input(4.1, Ted::Gardner, 3000, 30);
    let want = kernel_run::<C32>(&cfg, r, &input);
    let mut block = SymbolSyncBlock::<C32, VecStorage>::new(&cfg, r, &mut VecStorage).unwrap();
    let mut got = Vec::new();
    let mut rest: &[C32] = &input;
    let mut room = [C32::default(); 3];
    loop {
        let p = block.process(rest, &mut room).unwrap();
        assert!(p.produced <= 3);
        got.extend_from_slice(&room[..p.produced]);
        rest = &rest[p.consumed..];
        if p.produced < 3 {
            assert!(rest.is_empty(), "stopped with room left");
            break;
        }
    }
    common::same_bits(&want, &got, "three outputs per call");
    // No output room: nothing is consumed.
    let mut fresh = SymbolSyncBlock::<C32, VecStorage>::new(&cfg, r, &mut VecStorage).unwrap();
    let p = fresh.process(&input, &mut []).unwrap();
    assert_eq!(p.produced, 0);
    assert!(
        p.consumed <= 3,
        "consumed {} without output room",
        p.consumed
    );
    block.reset();
    assert_eq!(block.sps_estimate(), 4.1);
    let again = run_block(&mut block, &input, None);
    common::same_bits(&want, &again, "after reset");
}

#[test]
fn symbol_sync_timing_is_variable_with_the_exact_nominal_ratio() {
    let cases: [(f64, u64, Ratio, Rate); 5] = [
        (
            1000.0,
            4100,
            Ratio::new(10, 41).unwrap(),
            Rate::hz(1000).unwrap(),
        ),
        (
            1187.5,
            19_000,
            Ratio::new(1, 16).unwrap(),
            Rate::new(2375, 2).unwrap(),
        ),
        (
            2375.0,
            19_000,
            Ratio::new(1, 8).unwrap(),
            Rate::hz(2375).unwrap(),
        ),
        // Rounded to 1/1000 Hz.
        (
            1_187.500_4,
            19_000,
            Ratio::new(1, 16).unwrap(),
            Rate::new(2375, 2).unwrap(),
        ),
        (
            0.001,
            1,
            Ratio::new(1, 1000).unwrap(),
            Rate::new(1, 1000).unwrap(),
        ),
    ];
    for (sym, fs, ratio, out) in cases {
        let cfg = SymbolSyncConfig {
            symbol_rate_hz: sym,
            ..SymbolSyncConfig::default()
        };
        let r = Rate::hz(fs).unwrap();
        let block = SymbolSyncBlock::<f32, VecStorage>::new(&cfg, r, &mut VecStorage).unwrap();
        let t = block.timing();
        assert_eq!(t.ratio, ratio, "{sym} at {fs}");
        assert_eq!(t.kind, RatioKind::Variable);
        assert_eq!(t.delay, Delay::new(1, 1).unwrap());
        assert_eq!(
            SymbolSyncBlock::<f32, VecStorage>::rate_out(&cfg, r).unwrap(),
            out
        );
        assert_eq!(r.scale(ratio).unwrap(), out);
        // One output is guaranteed after the largest strobe step (1.25 omega at the upper
        // limit of omega) plus the interpolation window.
        let sps = cfg.sps(r).unwrap();
        assert_eq!(
            block.min_input(),
            (sps * (1.0 + cfg.max_deviation) * 1.25).ceil() as usize + 3
        );
    }
}

#[test]
fn symbol_sync_needs_two_samples_per_symbol_and_no_storage() {
    let ok = |sym: f64| SymbolSyncConfig {
        symbol_rate_hz: sym,
        ..SymbolSyncConfig::default()
    };
    // Exactly 2 samples per symbol is the limit.
    assert!(ok(24_000.0).validate(rate(48_000)).is_ok());
    for (sym, fs) in [(24_000.001, 48_000), (30_000.0, 48_000), (1e9, 48_000)] {
        let cfg = ok(sym);
        match SymbolSyncBlock::<C32, VecStorage>::new(&cfg, rate(fs), &mut VecStorage) {
            Err(Error::OutOfRange { name, max, .. }) => {
                assert_eq!(name, "symbol_rate_hz");
                assert_eq!(max, fs as f64 / 2.0);
            }
            other => panic!("{sym}: {other:?}"),
        }
    }
    // More than 10^6 samples per symbol is out of range too (D531).
    assert!(matches!(
        ok(0.001).validate(rate(2000)),
        Err(Error::OutOfRange {
            name: "symbol_rate_hz",
            ..
        })
    ));
    // Every parameter outside its range is OutOfRange with its name.
    let base = ok(1000.0);
    for (cfg, name) in [
        (
            SymbolSyncConfig {
                loop_bw: 0.2,
                ..base
            },
            "loop_bw",
        ),
        (
            SymbolSyncConfig {
                damping: 0.0,
                ..base
            },
            "damping",
        ),
        (
            SymbolSyncConfig {
                max_deviation: 0.06,
                ..base
            },
            "max_deviation",
        ),
        (
            SymbolSyncConfig {
                ted_gain: 0.0,
                ..base
            },
            "ted_gain",
        ),
        (
            SymbolSyncConfig {
                symbol_rate_hz: f64::NAN,
                ..base
            },
            "symbol_rate_hz",
        ),
    ] {
        assert!(
            matches!(cfg.validate(rate(8000)), Err(Error::OutOfRange { name: n, .. }) if n == name),
            "{name}"
        );
    }
    // No Q15 variant (D529).
    assert!(matches!(
        SymbolSyncBlock::<CQ15, VecStorage>::new(&base, rate(8000), &mut VecStorage),
        Err(Error::Core(sdrk::Error::Unsupported(_)))
    ));
    for (sym, fs) in [(1000.0, 4100), (24_000.0, 48_000)] {
        let cfg = ok(sym);
        let r = rate(fs);
        let bytes = SymbolSyncBlock::<C32, A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert_eq!(bytes, 0);
        check_exact_storage(bytes, |a| {
            SymbolSyncBlock::<C32, _>::new(&cfg, r, a).map(|_| ())
        });
        check_exact_storage(bytes, |a| {
            SymbolSyncBlock::<f32, _>::new(&cfg, r, a).map(|_| ())
        });
    }
}

// ---- slice ----------------------------------------------------------------------------------

/// Corner values of a component: zeros of both signs, the smallest subnormal, the extremes,
/// infinities and ordinary values.
const CORNERS: [f32; 10] = [
    0.0,
    -0.0,
    1.0e-45,
    -1.0e-45,
    f32::MAX,
    f32::MIN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    0.75,
    -1.5,
];

fn slice_block<T: sdrk_blocks::SliceElem>(
    c: SliceConstellation,
    scale: f64,
) -> SliceBlock<T, VecStorage> {
    let cfg = SliceConfig {
        constellation: c,
        scale,
    };
    SliceBlock::new(&cfg, rate(1000), &mut VecStorage).unwrap()
}

/// `f32` values equal bit for bit, NaN with NaN.
fn same_f32(want: &[f32], got: &[f32], what: &str) {
    assert_eq!(want.len(), got.len(), "{what}");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        assert!(
            w.to_bits() == g.to_bits() || (w.is_nan() && g.is_nan()),
            "{what} at {i}: {w} vs {g}"
        );
    }
}

#[test]
fn slice_follows_the_table_at_corner_values() {
    for scale in [1.0, 2.5, 1e-9, 1e9] {
        let s = scale as f32;
        // real: s x
        let mut b = slice_block::<f32>(SliceConstellation::Real, scale);
        let mut out = vec![7.0f32; CORNERS.len() + 1];
        let p = b.process(&CORNERS, &mut out).unwrap();
        assert_eq!((p.consumed, p.produced), (CORNERS.len(), CORNERS.len()));
        let want: Vec<f32> = CORNERS.iter().map(|x| s * x).collect();
        same_f32(&want, &out[..CORNERS.len()], "real");
        assert_eq!(out[CORNERS.len()], 7.0, "real wrote past the symbols");
        // bpsk: s re; qpsk: s re, s im
        let symbols: Vec<C32> = CORNERS
            .iter()
            .flat_map(|&re| CORNERS.iter().map(move |&im| C32::new(re, im)))
            .collect();
        let mut b = slice_block::<C32>(SliceConstellation::Bpsk, scale);
        let mut out = vec![0.0f32; symbols.len()];
        let p = b.process(&symbols, &mut out).unwrap();
        assert_eq!((p.consumed, p.produced), (symbols.len(), symbols.len()));
        let want: Vec<f32> = symbols.iter().map(|c| s * c.re).collect();
        same_f32(&want, &out, "bpsk");
        let mut b = slice_block::<C32>(SliceConstellation::Qpsk, scale);
        let mut out = vec![0.0f32; 2 * symbols.len()];
        let p = b.process(&symbols, &mut out).unwrap();
        assert_eq!((p.consumed, p.produced), (symbols.len(), 2 * symbols.len()));
        let want: Vec<f32> = symbols.iter().flat_map(|c| [s * c.re, s * c.im]).collect();
        same_f32(&want, &out, "qpsk");
    }
    // NaN passes through as NaN.
    let mut b = slice_block::<C32>(SliceConstellation::Qpsk, 1.0);
    let mut out = [0.0f32; 2];
    b.process(&[C32::new(f32::NAN, -1.0)], &mut out).unwrap();
    assert!(out[0].is_nan());
    assert_eq!(out[1], -1.0);
}

/// The QPSK bit order: bit 0 from `re`, bit 1 from `im` (Gray), so the four quadrants give
/// the bit pairs 11, 01, 00, 10 (bit 0 first), with `> 0` meaning 1.
#[test]
fn slice_qpsk_bit_order() {
    let quadrants = [
        C32::new(0.7, 0.7),
        C32::new(-0.7, 0.7),
        C32::new(-0.7, -0.7),
        C32::new(0.7, -0.7),
    ];
    let mut b = slice_block::<C32>(SliceConstellation::Qpsk, 1.0);
    let mut out = [0.0f32; 8];
    b.process(&quadrants, &mut out).unwrap();
    let bits: Vec<u8> = out.iter().map(|&x| u8::from(x > 0.0)).collect();
    assert_eq!(bits, [1, 1, 0, 1, 0, 0, 1, 0]);
    // An odd output room takes only whole symbols.
    let mut b = slice_block::<C32>(SliceConstellation::Qpsk, 1.0);
    let mut out = [9.0f32; 5];
    let p = b.process(&quadrants, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (2, 4));
    assert_eq!(out[4], 9.0);
    let p = b.process(&quadrants, &mut out[..1]).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
}

#[test]
fn slice_scale_is_a_runtime_parameter() {
    let mut b = slice_block::<C32>(SliceConstellation::Bpsk, 1.0);
    let mut out = [0.0f32; 1];
    b.process(&[C32::new(0.5, 3.0)], &mut out).unwrap();
    assert_eq!(out[0], 0.5);
    b.set_param("scale", ParamValue::F64(8.0)).unwrap();
    assert_eq!(b.param("scale"), Ok(ParamValue::F64(8.0)));
    b.process(&[C32::new(0.5, 3.0)], &mut out).unwrap();
    assert_eq!(out[0], 4.0);
    assert!(matches!(
        b.set_param("scale", ParamValue::F64(0.0)),
        Err(Error::OutOfRange { name: "scale", .. })
    ));
    assert_eq!(
        b.set_param("constellation", ParamValue::Choice(2)),
        Err(Error::ReadOnlyParameter)
    );
    // reset changes nothing (the block has no state).
    b.reset();
    b.process(&[C32::new(0.5, 3.0)], &mut out).unwrap();
    assert_eq!(out[0], 4.0);
}

#[test]
fn slice_timing_rate_storage_and_types() {
    let r = rate(1200);
    for (c, bits) in [
        (SliceConstellation::Real, 1u64),
        (SliceConstellation::Bpsk, 1),
        (SliceConstellation::Qpsk, 2),
    ] {
        let cfg = SliceConfig {
            constellation: c,
            scale: 1.0,
        };
        let t = if c == SliceConstellation::Real {
            let b = SliceBlock::<f32, VecStorage>::new(&cfg, r, &mut VecStorage).unwrap();
            assert!(SliceBlock::<C32, VecStorage>::new(&cfg, r, &mut VecStorage).is_err());
            assert_eq!(
                SliceBlock::<f32, VecStorage>::rate_out(&cfg, r).unwrap(),
                rate(1200 * bits)
            );
            check_exact_storage(0, |a| SliceBlock::<f32, A<'_>>::new(&cfg, r, a).map(|_| ()));
            b.timing()
        } else {
            let b = SliceBlock::<C32, VecStorage>::new(&cfg, r, &mut VecStorage).unwrap();
            assert!(matches!(
                SliceBlock::<f32, VecStorage>::new(&cfg, r, &mut VecStorage),
                Err(Error::InvalidParameter {
                    name: "constellation",
                    ..
                })
            ));
            assert_eq!(
                SliceBlock::<C32, VecStorage>::rate_out(&cfg, r).unwrap(),
                rate(1200 * bits)
            );
            check_exact_storage(0, |a| SliceBlock::<C32, A<'_>>::new(&cfg, r, a).map(|_| ()));
            b.timing()
        };
        assert_eq!(t.ratio, Ratio::new(bits, 1).unwrap());
        assert_eq!(t.kind, RatioKind::Constant);
        assert_eq!(t.delay, Delay::ZERO);
        assert_eq!(
            SliceBlock::<C32, VecStorage>::storage_bytes(
                &SliceConfig {
                    constellation: SliceConstellation::Qpsk,
                    scale: 1.0
                },
                r
            ),
            Ok(0)
        );
    }
}

/// `slice` after `symbol_sync` in the cut plans: the soft bits of a chunked run equal the
/// whole run (the slicer has no state, the synchronizer is block-size invariant).
#[test]
fn symbol_sync_then_slice_is_block_size_invariant() {
    let cfg = SymbolSyncConfig {
        symbol_rate_hz: 1000.0,
        ..SymbolSyncConfig::default()
    };
    let r = rate(4100);
    let input = c32_input(4.1, Ted::Gardner, 4000, 40);
    let symbols = kernel_run::<C32>(&cfg, r, &input);
    let want: Vec<f32> = symbols.iter().map(|c| 3.0 * c.re).collect();
    check_block(
        &want,
        &symbols,
        SEED,
        || slice_block::<C32>(SliceConstellation::Bpsk, 3.0),
        "slice bpsk",
    );
}
