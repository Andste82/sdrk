//! The blocks of M44: the `slice` constellations `qam16` and `qam64` (test 1: the soft-bit
//! signs of every point derived from the mapping of `sdrk_core::kernels::constellation`, the
//! piecewise-linear max-log values bit for bit, the agreement of every sign with the grid
//! decision, the rates, the parameter checks, the extremes and 20 cut plans of seed 607), and the
//! blocks `costas` and `equalize` with the constellations `qam16` and `qam64` (M44 §3: the block
//! equals the kernel whole and in 20 cut plans of seed 607, the final state, `reset`, lock and
//! frequency; the constellation choice). M44 §4: `qam_mod` and `qam_demod` equal their
//! kernels whole and in 20 cut plans of seed 607 (test 8), their timing, rates and parameters,
//! a noiseless link without bit errors, and `frame_decode` with `rotations` `four` (test 1: the
//! sync word and the payload under each of the four rotations for 2, 4 and 6 bits per symbol,
//! the rotation in the packet flags, cut plans, the parameter checks, and a QAM link turned by
//! 0 to 270 degrees).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::approx_constant
)]

mod common;

use common::{check_block, rate, run_block, same_bits, twenty_plans};
use sdrk::{C32, Delay, Ratio, RatioKind, Sdrk};
use sdrk_blocks::design::{rrc, rrc_len};
use sdrk_blocks::{
    Block, CostasBlock, CostasConfig, CostasConstellation, EqConstellation, EqMode, EqualizeBlock,
    EqualizeConfig, Error, FrameDecodeBlock, FrameDecodeConfig, FrameRotations, Packet,
    PacketFlags, ParamValue, PortType, QAM_ACQUISITION_CHECK, QAM_ACQUISITION_SKIP,
    QAM_ACQUISITION_SYMBOLS, QamDemodBlock, QamDemodConfig, QamModBlock, QamModConfig, SliceBlock,
    SliceConfig, SliceConstellation,
};
use sdrk_core::kernels::constellation::{Qam, rotate_symbol};
use sdrk_core::kernels::costas::{Costas, CostasState};
use sdrk_core::kernels::equalize::{EqParams, Equalize, EqualizeState};
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, Scaling};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::interp::{FirInterpState, FirInterpolate};
use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncState, Ted};
use sdrk_proto::frame::{ROTATION_SHIFT, Whitening, encode, frame_bits, rotate_group, rotation_of};
use sdrk_testkit::{Pcg32, VecStorage};

/// The seed of the cut plans and signals of M44 (test 8).
const SEED: u64 = 607;

const QAMS: [(SliceConstellation, Qam); 2] = [
    (SliceConstellation::Qam16, Qam::Qam16),
    (SliceConstellation::Qam64, Qam::Qam64),
];

fn slicer(constellation: SliceConstellation, scale: f64) -> SliceBlock<C32, VecStorage> {
    let cfg = SliceConfig {
        constellation,
        scale,
    };
    SliceBlock::<C32, _>::new(&cfg, rate(25_000), &mut VecStorage).unwrap()
}

fn slice_all(constellation: SliceConstellation, scale: f64, input: &[C32]) -> Vec<f32> {
    let bits = constellation.bits_per_symbol();
    let mut out = vec![0.0f32; bits * input.len()];
    let p = slicer(constellation, scale)
        .process(input, &mut out)
        .unwrap();
    assert_eq!((p.consumed, p.produced), (input.len(), bits * input.len()));
    out
}

/// The `f32` definition of M44 §2 per axis: `v = x √E`, then `s v`, `s (2 - |v|)` (16-QAM) or
/// `s v`, `s (4 - |v|)`, `s (2 - ||v| - 4|)` (64-QAM).
fn axis_bits(q: Qam, s: f32, x: f32) -> Vec<f32> {
    let v = x * q.sqrt_energy();
    let a = v.abs();
    match q {
        Qam::Qam16 => vec![s * v, s * (2.0 - a)],
        _ => vec![s * v, s * (4.0 - a), s * (2.0 - (a - 4.0).abs())],
    }
}

/// `a` and `b` bit for bit, every NaN folded to one value: NaN must be NaN where the
/// definition gives NaN, its sign and payload are not part of the contract (D296; the SIMD paths
/// of the kernel `qam_soft_bits` may set the sign of a NaN, D1414).
fn same_f32_bits(a: &[f32], b: &[f32], what: &str) {
    let bits = |x: &[f32]| {
        x.iter()
            .map(|v| if v.is_nan() { u32::MAX } else { v.to_bits() })
            .collect::<Vec<_>>()
    };
    same_bits(&bits(a), &bits(b), what);
}

fn reference(q: Qam, s: f32, input: &[C32]) -> Vec<f32> {
    input
        .iter()
        .flat_map(|y| {
            let mut b = axis_bits(q, s, y.re);
            b.extend(axis_bits(q, s, y.im));
            b
        })
        .collect()
}

#[test]
fn the_qam_constellations_take_c32_and_give_their_bits() {
    for (c, q) in QAMS {
        assert_eq!(c.bits_per_symbol(), q.bits_per_symbol());
        assert_eq!(c.input_type(), PortType::C32);
        let cfg = SliceConfig {
            constellation: c,
            scale: 1.0,
        };
        assert_eq!(
            SliceBlock::<C32, VecStorage>::rate_out(&cfg, rate(25_000)).unwrap(),
            rate(25_000 * q.bits_per_symbol() as u64)
        );
        assert!(SliceBlock::<f32, _>::new(&cfg, rate(25_000), &mut VecStorage).is_err());
        let t = slicer(c, 1.0).timing();
        assert_eq!(t.ratio, Ratio::new(q.bits_per_symbol() as u64, 1).unwrap());
        assert_eq!((t.kind, t.delay), (RatioKind::Constant, Delay::ZERO));
    }
    let mut cfg = SliceConfig::default();
    // The options are appended: qam16 is choice 5, qam64 choice 6.
    for (i, c) in [
        (5u16, SliceConstellation::Qam16),
        (6, SliceConstellation::Qam64),
    ] {
        cfg.set("constellation", ParamValue::Choice(i)).unwrap();
        assert_eq!(cfg.constellation, c);
        assert_eq!(cfg.get("constellation").unwrap(), ParamValue::Choice(i));
    }
}

/// M44 test 1: every point slices to its own bits, the signs derived from the mapping of
/// `Qam::point` (bit `j` of the symbol, most significant first, is 1 when soft bit `j` is above
/// 0), and the values are the `f32` definition bit for bit.
#[test]
fn every_point_gives_the_signs_of_its_mapping() {
    for (c, q) in QAMS {
        let k = q.bits_per_symbol();
        let points: Vec<C32> = (0..q.points() as u32).map(|s| q.point(s)).collect();
        for scale in [1.0, 0.25, 3.0] {
            let out = slice_all(c, scale, &points);
            same_bits(&reference(q, scale as f32, &points), &out, "points");
            for (s, bits) in out.chunks_exact(k).enumerate() {
                for (j, &b) in bits.iter().enumerate() {
                    let bit = (s >> (k - 1 - j)) & 1 == 1;
                    assert_eq!(b > 0.0, bit, "{q:?} symbol {s} bit {j}: {b}");
                    // On a point every soft bit is at least one grid unit from the boundary.
                    assert!(
                        b.abs() >= 0.99 * scale as f32,
                        "{q:?} symbol {s} bit {j}: {b}"
                    );
                }
            }
        }
    }
}

/// The sign of every soft bit is the bit of the grid decision of `Qam::decide_symbol` (away
/// from the decision boundaries, where the `f32` rounding of the two forms may differ), on
/// random symbols of seed 607; the values are the `f32` definition bit for bit, whole and in
/// 20 cut plans.
#[test]
fn the_signs_agree_with_the_grid_decision() {
    let mut rng = Pcg32::new(607, 1);
    for (c, q) in QAMS {
        let k = q.bits_per_symbol();
        let input: Vec<C32> = (0..4000)
            .map(|_| C32::new(0.7 * rng.normal() as f32, 0.7 * rng.normal() as f32))
            .collect();
        let out = slice_all(c, 1.0, &input);
        same_bits(&reference(q, 1.0, &input), &out, "random");
        let near = |x: f32| {
            let v = x * q.sqrt_energy();
            (v.round() as i32) % 2 == 0 && (v - v.round()).abs() < 1e-4
        };
        let mut checked = 0;
        for (y, bits) in input.iter().zip(out.chunks_exact(k)) {
            if near(y.re) || near(y.im) {
                continue;
            }
            let d = q.decide_symbol(*y) as usize;
            for (j, &b) in bits.iter().enumerate() {
                assert_eq!(b > 0.0, (d >> (k - 1 - j)) & 1 == 1, "{q:?} {y:?} bit {j}");
            }
            checked += 1;
        }
        assert!(checked > 3900, "{checked}");
        let want = out.clone();
        check_block(&want, &input, 607, || slicer(c, 1.0), "qam slice cut plans");
    }
}

/// Silence, full scale, NaN and the infinities: zero gives finite soft bits, NaN gives NaN,
/// an infinite component gives infinite soft bits of its axis and leaves the other axis alone.
#[test]
fn the_qam_slicer_handles_the_extremes() {
    for (c, q) in QAMS {
        let k = q.bits_per_symbol();
        let h = k / 2;
        let input = [
            C32::new(0.0, 0.0),
            C32::new(1.0, -1.0),
            C32::new(f32::NAN, 0.5),
            C32::new(f32::INFINITY, f32::NEG_INFINITY),
            C32::new(f32::MAX, 0.0),
        ];
        let out = slice_all(c, 1.0, &input);
        same_f32_bits(&reference(q, 1.0, &input), &out, "extremes");
        assert!(out[..2 * k].iter().all(|b| b.is_finite()));
        let nan = &out[2 * k..3 * k];
        assert!(nan[..h].iter().all(|b| b.is_nan()));
        assert!(nan[h..].iter().all(|b| b.is_finite()));
        assert!(out[3 * k..4 * k].iter().all(|b| b.is_infinite()));
        assert_eq!(out[3 * k], f32::INFINITY);
        assert_eq!(out[3 * k + h], f32::NEG_INFINITY);
        assert!(out[4 * k..].iter().all(|b| !b.is_nan()));
        // A short output takes only the symbols that fit.
        let mut short = vec![0.0f32; 2 * k + 1];
        let p = slicer(c, 1.0).process(&input, &mut short).unwrap();
        assert_eq!((p.consumed, p.produced), (2, 2 * k));
    }
}

/// Random points of `q` through `1 + h1 z^-1` (`h1` 0 for none), turned by `phase0 + 2 pi
/// offset k`, plus uniform noise of `noise` per component.
fn qam_signal(
    q: Qam,
    n: usize,
    h1: C32,
    phase0: f32,
    offset: f32,
    noise: f32,
    stream: u64,
) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED, stream);
    let s: Vec<C32> = (0..n).map(|_| q.point(rng.next_u32())).collect();
    (0..n)
        .map(|k| {
            let v = if k > 0 { s[k] + s[k - 1] * h1 } else { s[k] };
            let (sn, cs) = (phase0 + core::f32::consts::TAU * offset * k as f32).sin_cos();
            C32::new(
                v.re * cs - v.im * sn + noise * rng.normal() as f32,
                v.re * sn + v.im * cs + noise * rng.normal() as f32,
            )
        })
        .collect()
}

#[test]
fn costas_block_with_qam_equals_the_kernel_whole_and_in_cut_plans() {
    for (stream, (q, c)) in [
        (Qam::Qam16, CostasConstellation::Qam16),
        (Qam::Qam64, CostasConstellation::Qam64),
    ]
    .into_iter()
    .enumerate()
    {
        let cfg = CostasConfig {
            constellation: c,
            loop_bw: 0.02,
            ..CostasConfig::default()
        };
        let noise = if q == Qam::Qam16 { 0.02 } else { 0.008 };
        let input = qam_signal(
            q,
            6000,
            C32::new(0.0, 0.0),
            0.3,
            0.0005,
            noise,
            stream as u64,
        );
        let mut st = CostasState::new(c, 0.02, 0.7071, 0.05).unwrap();
        let mut want = vec![C32::default(); input.len()];
        <Sdrk as Costas>::costas(&Sdrk, &mut st, &input, &mut want).unwrap();
        check_block(
            &want,
            &input,
            SEED,
            || CostasBlock::<VecStorage>::new(&cfg, rate(25_000), &mut VecStorage).unwrap(),
            &format!("costas {q:?}"),
        );
        let mut block =
            CostasBlock::<VecStorage>::new(&cfg, rate(25_000), &mut VecStorage).unwrap();
        let got = run_block(&mut block, &input, None);
        same_bits(&want, &got, &format!("costas {q:?}"));
        assert_eq!(block.state(), &st);
        assert!(block.locked(), "{q:?}");
        assert!(
            (block.frequency() - 0.0005).abs() < 2e-4,
            "{q:?}: {}",
            block.frequency()
        );
        // the tail sits on the grid (the loop started within the quarter of the points)
        for y in &got[4000..] {
            let d = q.decide(*y);
            assert!((y.re - d.re).hypot(y.im - d.im) < 0.1, "{q:?} {y}");
        }
        block.reset();
        assert_eq!(
            block.state(),
            &CostasState::new(c, 0.02, 0.7071, 0.05).unwrap()
        );
        assert_eq!(
            block.param("constellation"),
            Ok(ParamValue::Choice(if q == Qam::Qam16 { 3 } else { 4 }))
        );
    }
}

#[test]
fn equalize_block_with_qam_equals_the_kernel_whole_and_in_cut_plans() {
    for (stream, (q, c)) in [
        (Qam::Qam16, EqConstellation::Qam16),
        (Qam::Qam64, EqConstellation::Qam64),
    ]
    .into_iter()
    .enumerate()
    {
        let cfg = EqualizeConfig {
            constellation: c,
            mu_cma: 0.02,
            mu_dd: 0.01,
            cma_symbols: 3000,
            ..EqualizeConfig::default()
        };
        let noise = if q == Qam::Qam16 { 0.01 } else { 0.003 };
        let input = qam_signal(
            q,
            8000,
            C32::new(0.2, 0.1),
            0.15,
            0.0,
            noise,
            10 + stream as u64,
        );
        let mut st = EqualizeState::new(
            vec![C32::default(); 11],
            vec![C32::default(); 11],
            EqParams {
                constellation: c,
                mu_cma: 0.02,
                mu_dd: 0.01,
                cma_symbols: 3000,
                ..EqParams::default()
            },
        )
        .unwrap();
        let mut want = vec![C32::default(); input.len()];
        <Sdrk as Equalize>::equalize(&Sdrk, &mut st, &input, &mut want).unwrap();
        check_block(
            &want,
            &input,
            SEED,
            || EqualizeBlock::<VecStorage>::new(&cfg, rate(25_000), &mut VecStorage).unwrap(),
            &format!("equalize {q:?}"),
        );
        let mut block =
            EqualizeBlock::<VecStorage>::new(&cfg, rate(25_000), &mut VecStorage).unwrap();
        let got = run_block(&mut block, &input, None);
        same_bits(&want, &got, &format!("equalize {q:?}"));
        same_bits(
            st.weights(),
            block.weights(),
            &format!("equalize {q:?} weights"),
        );
        assert_eq!(block.mode(), EqMode::Decision);
        assert!(block.mse() < 0.01, "{q:?}: mse {}", block.mse());
        assert_eq!(block.state().constellation().cma_r2(), c.cma_r2());
        assert_eq!(
            block.param("constellation"),
            Ok(ParamValue::Choice(if q == Qam::Qam16 { 3 } else { 4 }))
        );
        assert!(
            got[7000..]
                .iter()
                .all(|y| y.re.is_finite() && y.im.is_finite())
        );
    }
}

// ---- qam_mod, qam_demod (M44 §4, test 8) -------------------------------------------------

/// Random soft bits `+-1` (stream `stream` of seed 607).
fn random_bits(n: usize, stream: u64) -> Vec<f32> {
    let mut rng = Pcg32::new(SEED, stream);
    (0..n)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

/// The point of the bits `b` (above 0 is a 1) written out from M44 §1 in `f64` and rounded once:
/// the first half the in-phase level index by its Gray code, the second half the quadrature one,
/// the level `2 i - (m - 1)` over `sqrt(2 (M - 1) / 3)`.
fn point_of_bits(b: &[f32]) -> C32 {
    let half = b.len() / 2;
    let m = 1usize << half;
    let level = |bits: &[f32]| {
        let g = bits
            .iter()
            .fold(0usize, |g, &x| (g << 1) | usize::from(x > 0.0));
        let i = (0..m).find(|&i| i ^ (i >> 1) == g).unwrap();
        let e = (2.0 * ((m * m) as f64 - 1.0) / 3.0).sqrt();
        ((2 * i) as f64 - (m - 1) as f64) / e
    };
    C32::new(level(&b[..half]) as f32, level(&b[half..]) as f32)
}

fn rrc_taps(rolloff: f64, sps: usize, span: usize, gain: f64) -> Vec<f32> {
    let mut h = vec![0.0f64; rrc_len(sps, span).unwrap()];
    rrc(rolloff, sps, span, &mut h).unwrap();
    h.iter().map(|v| (v * gain) as f32).collect()
}

/// `qam_mod` composed from the kernel `fir_interpolate` in one call on the points of
/// [`point_of_bits`].
fn mod_kernels(cfg: &QamModConfig, bits: &[f32]) -> Vec<C32> {
    let sps = cfg.samples_per_symbol;
    let taps = rrc_taps(cfg.rolloff, sps, cfg.span, (sps as f64).sqrt());
    let n = taps.len();
    let mut st =
        FirInterpState::<C32, _, _>::new(taps, vec![C32::default(); n.div_ceil(sps)], sps).unwrap();
    let syms: Vec<C32> = bits
        .chunks_exact(cfg.bits_per_symbol())
        .map(point_of_bits)
        .collect();
    let mut out = vec![C32::default(); syms.len() * sps];
    let p =
        <Sdrk as FirInterpolate<C32>>::fir_interpolate(&Sdrk, &mut st, &syms, &mut out).unwrap();
    assert_eq!(p.consumed, syms.len());
    out
}

#[test]
fn qam_mod_equals_the_kernels_whole_and_in_cut_plans() {
    for (i, (order, sps, span)) in [(16u32, 4usize, 10usize), (64, 8, 6), (16, 2, 32)]
        .into_iter()
        .enumerate()
    {
        let cfg = QamModConfig {
            order,
            samples_per_symbol: sps,
            span,
            ..QamModConfig::default()
        };
        let k = cfg.bits_per_symbol();
        let bits = random_bits(1500 * k, 10 + i as u64);
        // The mapping of M44 §1 equals the table of `Qam` bit for bit.
        let q = if order == 16 { Qam::Qam16 } else { Qam::Qam64 };
        for (s, b) in bits.chunks_exact(k).enumerate().take(200) {
            let sym = b.iter().fold(0u32, |g, &x| (g << 1) | u32::from(x > 0.0));
            assert_eq!(point_of_bits(b), q.point(sym), "{order} symbol {s}");
        }
        let want = mod_kernels(&cfg, &bits);
        let make = || QamModBlock::<VecStorage>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
        check_block(&want, &bits, SEED, make, &format!("qam_mod case {i}"));
        // reset starts over
        let mut block = make();
        let first = run_block(&mut block, &bits, None);
        block.reset();
        same_bits(&first, &run_block(&mut block, &bits, None), "qam_mod reset");
    }
}

#[test]
fn qam_mod_timing_rates_and_parameters() {
    let cfg = QamModConfig {
        order: 64,
        samples_per_symbol: 8,
        ..QamModConfig::default()
    };
    let fs = rate(150_000);
    let block = QamModBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
    assert_eq!(block.num_taps(), 81);
    let t = block.timing();
    assert_eq!(t.ratio, Ratio::new(8, 6).unwrap());
    assert_eq!(t.kind, RatioKind::Constant);
    assert_eq!(t.delay, Delay::new(60, 2).unwrap()); // span 10 symbols of 6 bits, halved
    assert_eq!(block.min_input(), 6);
    assert_eq!(
        QamModBlock::<VecStorage>::rate_out(&cfg, fs).unwrap(),
        rate(200_000)
    );
    // whole symbols only: 11 bits of 16-QAM give two symbols
    let cfg16 = QamModConfig {
        samples_per_symbol: 4,
        ..QamModConfig::default()
    };
    let mut m = QamModBlock::<VecStorage>::new(&cfg16, fs, &mut VecStorage).unwrap();
    let mut out = vec![C32::default(); 64];
    let p = m.process(&random_bits(11, 3), &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (8, 8));
    // room for one symbol only
    let p = m.process(&random_bits(8, 4), &mut out[..7]).unwrap();
    assert_eq!((p.consumed, p.produced), (4, 4));
    // the orders 17 to 63 are no orders; outside 16 to 64 out of range
    let mut c = QamModConfig::default();
    assert!(matches!(
        c.set("order", ParamValue::I64(32)),
        Err(Error::InvalidParameter { name: "order", .. })
    ));
    assert!(matches!(
        c.set("order", ParamValue::I64(4)),
        Err(Error::OutOfRange { name: "order", .. })
    ));
    c.set("order", ParamValue::I64(64)).unwrap();
    assert_eq!(c.get("order"), Ok(ParamValue::I64(64)));
    assert!(QamModConfig { span: 3, ..c }.validate().is_err());
    assert!(
        QamModConfig {
            samples_per_symbol: 65,
            ..c
        }
        .validate()
        .is_err()
    );
    let d = sdrk_blocks::all_descriptions()
        .iter()
        .find(|d| d.kind == "qam_mod")
        .unwrap();
    assert_eq!(d.inputs[0].types, [PortType::F32]);
    assert_eq!(d.outputs[0].types, [PortType::C32]);
    assert_eq!(d.required, ["samples_per_symbol"]);
    assert_eq!(
        m.set_param("order", ParamValue::I64(64)),
        Err(Error::ReadOnlyParameter)
    );
}

/// A noisy, turned and frequency-shifted QAM signal from [`mod_kernels`] at `sps` samples per
/// symbol: the phase `0.6 + 2 pi 0.001 k / sps` at sample `k` and Gaussian noise of `noise`
/// per component (streams of seed 607).
fn received(order: u32, sps: usize, n_sym: usize, noise: f64, stream: u64) -> Vec<C32> {
    let cfg = QamModConfig {
        order,
        samples_per_symbol: sps,
        ..QamModConfig::default()
    };
    let tx = mod_kernels(&cfg, &random_bits(n_sym * cfg.bits_per_symbol(), stream));
    let mut rng = Pcg32::new(SEED, 100 + stream);
    tx.iter()
        .enumerate()
        .map(|(i, x)| {
            let ph = 0.6 + 2.0 * std::f64::consts::PI * 0.001 * i as f64 / sps as f64;
            let (s, c) = ph.sin_cos();
            let (re, im) = (f64::from(x.re), f64::from(x.im));
            C32::new(
                (re * c - im * s + noise * rng.normal()) as f32,
                (re * s + im * c + noise * rng.normal()) as f32,
            )
        })
        .collect()
}

/// The symbols of the frequency acquisition of `qam_demod`: skipped after the start, then
/// collected, and the FFT size of the estimate.
const ACQ_SKIP: usize = 256;
const ACQ_SYMBOLS: usize = 512;
const ACQ_FFT: usize = 1024;

/// The frequency acquisition of `qam_demod` written out (M44, I44-3, I44-4) over the symbols
/// `x` of the timing loop: the smoothed power `p = p + (|x|^2 - p) / 64` (from 0, finite
/// values only), and from symbol 256 on windows of 512 symbols of `q = x^4 / (p p)` (`x^4` as
/// `(x^2)^2`, both parts divided by `p p`, in `f32`; 0 when `p` is not above 0 or `q` is not
/// finite), zero-padded to 1024 points, the kernel `fft`, the largest bin `k` (the first of
/// equal ones), used when `|X_k|^2 >= 12 sum |q|^2`, refined by the parabola through the
/// magnitudes of the bins `k - 1`, `k`, `k + 1`, giving
/// `((k + delta) / 1024 wrapped into [-0.5, 0.5)) / 4` cycles per symbol; a window without a
/// clear peak is followed by the next one. Returns the number of symbols after which the
/// estimate was used, and the estimate.
fn acquire_kernels(x: &[C32]) -> Option<(usize, f64)> {
    let plan = FftPlan::new(ACQ_FFT, vec![C32::default(); ACQ_FFT / 2]).unwrap();
    let mut p = 0.0f32;
    let mut buf = vec![C32::default(); ACQ_FFT];
    let mut fill = 0;
    for (n, v) in x.iter().enumerate() {
        let e = v.re * v.re + v.im * v.im;
        if e.is_finite() {
            p += (e - p) / 64.0;
            if p.abs() < f32::MIN_POSITIVE {
                p = 0.0;
            }
        }
        if n < ACQ_SKIP {
            continue;
        }
        let a = C32::new(v.re * v.re - v.im * v.im, 2.0 * v.re * v.im);
        let pp = p * p;
        let q = C32::new((a.re * a.re - a.im * a.im) / pp, (2.0 * a.re * a.im) / pp);
        let ok = p > 0.0 && q.re.is_finite() && q.im.is_finite();
        buf[fill] = if ok { q } else { C32::default() };
        fill += 1;
        if fill < ACQ_SYMBOLS {
            continue;
        }
        fill = 0;
        let energy: f32 = buf[..ACQ_SYMBOLS]
            .iter()
            .map(|v| v.re * v.re + v.im * v.im)
            .sum();
        buf[ACQ_SYMBOLS..].fill(C32::default());
        <Sdrk as Fft<C32>>::fft(&Sdrk, &plan, &mut buf, Direction::Forward, Scaling::Default)
            .unwrap();
        let power: Vec<f32> = buf.iter().map(|v| v.re * v.re + v.im * v.im).collect();
        let mut k = 0;
        for i in 1..ACQ_FFT {
            if power[i] > power[k] {
                k = i;
            }
        }
        if !(energy.is_finite() && energy > 0.0 && power[k] >= 12.0 * energy) {
            continue;
        }
        let mag = |i: usize| f64::from(power[i % ACQ_FFT]).sqrt();
        let (a, b, c) = (mag(k + ACQ_FFT - 1), mag(k), mag(k + 1));
        let den = a - 2.0 * b + c;
        let delta = if den < 0.0 { 0.5 * (a - c) / den } else { 0.0 };
        let mut w = (k as f64 + delta) / ACQ_FFT as f64;
        if w >= 0.5 {
            w -= 1.0;
        }
        return Some((n + 1, w / 4.0));
    }
    None
}

/// `qam_demod` composed from the kernels: the matched filter, `symbol_sync` (Gardner at the
/// exact samples per symbol), `equalize` (the QAM constellation, no feedback, `mu_cma` 0.05,
/// `mu_dd` 0.02, 1000 CMA symbols, the tracker at `carrier_bw`), `costas` with the QAM
/// constellation, and the soft bits of M44 §2 in `f32` ([`axis_bits`]); the equalizer and the
/// carrier loop run in two calls split after the symbol at which the frequency acquisition
/// ([`acquire_kernels`]) used its estimate `f`, which then goes to the equalizer's tracker (and
/// the carrier loop's frequency to 0), or without the equalizer to the carrier loop.
fn demod_kernels(cfg: &QamDemodConfig, fs: sdrk::Rate, input: &[C32]) -> Vec<f32> {
    let q = if cfg.order == 16 {
        Qam::Qam16
    } else {
        Qam::Qam64
    };
    let n = (fs.as_f64() / cfg.symbol_rate_hz).round() as usize;
    let taps = rrc_taps(cfg.rolloff, n, cfg.span, 1.0 / (n as f64).sqrt());
    let len = taps.len();
    let mut fir = FirState::<C32, _, _>::new(taps, vec![C32::default(); len - 1]).unwrap();
    let mut mf = vec![C32::default(); input.len()];
    <Sdrk as Fir<C32>>::fir(&Sdrk, &mut fir, input, &mut mf).unwrap();
    let sps = sdrk_blocks::SymbolSyncConfig {
        symbol_rate_hz: cfg.symbol_rate_hz,
        ..sdrk_blocks::SymbolSyncConfig::default()
    }
    .sps(fs)
    .unwrap();
    let mut sync =
        SymbolSyncState::<C32>::new(sps, Ted::Gardner, cfg.sync_bw, 0.7071, 0.005, 1.0).unwrap();
    let mut syms = vec![C32::default(); input.len()];
    let p = <Sdrk as SymbolSync<C32>>::symbol_sync(&Sdrk, &mut sync, &mf, &mut syms).unwrap();
    assert_eq!(p.consumed, input.len());
    syms.truncate(p.produced);
    let t = cfg.equalizer_taps;
    let mut eq = (t != 0).then(|| {
        EqualizeState::new(
            vec![C32::default(); t],
            vec![C32::default(); t],
            EqParams {
                constellation: if q == Qam::Qam16 {
                    EqConstellation::Qam16
                } else {
                    EqConstellation::Qam64
                },
                feedback_taps: 0,
                mu_cma: 0.05,
                mu_dd: 0.02,
                cma_symbols: 1000,
                phase_bw: cfg.carrier_bw,
            },
        )
        .unwrap()
    });
    let c = if q == Qam::Qam16 {
        CostasConstellation::Qam16
    } else {
        CostasConstellation::Qam64
    };
    let mut st = CostasState::new(c, cfg.carrier_bw, 0.7071, 0.05).unwrap();
    let acquired = acquire_kernels(&syms);
    let split = acquired.map_or(syms.len(), |(n, _)| n);
    // The input ends before the first check of the lock detector after the estimate, so the
    // restart of the block at a check without lock (D1421) is not part of this definition;
    // `qam_demod_searches_again_when_the_loop_does_not_lock` tests it.
    assert!(acquired.is_none_or(|(n, _)| syms.len() < n + QAM_ACQUISITION_CHECK));
    let mut y = vec![C32::default(); syms.len()];
    for (from, to) in [(0, split), (split, syms.len())] {
        let mut e = syms[from..to].to_vec();
        if let Some(eq) = eq.as_mut() {
            <Sdrk as Equalize>::equalize(&Sdrk, eq, &syms[from..to], &mut e).unwrap();
        }
        <Sdrk as Costas>::costas(&Sdrk, &mut st, &e, &mut y[from..to]).unwrap();
        if let Some((n, f)) = acquired.filter(|&(n, _)| n == to) {
            assert_eq!(n, split);
            match eq.as_mut() {
                Some(eq) => {
                    eq.set_frequency(f as f32);
                    st.set_frequency(0.0);
                }
                None => st.set_frequency(f),
            }
        }
    }
    reference(q, 1.0, &y)
}

#[test]
fn qam_demod_equals_the_kernels_whole_and_in_cut_plans() {
    for (i, (order, sps, taps)) in [
        (16u32, 4usize, 11usize),
        (64, 5, 15),
        (16, 2, 0),
        (64, 8, 0),
    ]
    .into_iter()
    .enumerate()
    {
        let fs = rate(9600 * sps as u64);
        let cfg = QamDemodConfig {
            order,
            symbol_rate_hz: 9600.0,
            equalizer_taps: taps,
            ..QamDemodConfig::default()
        };
        let k = cfg.bits_per_symbol();
        let input = received(order, sps, 3000, 0.02, 20 + i as u64);
        let want = demod_kernels(&cfg, fs, &input);
        assert!(want.len() > 2900 * k, "case {i}");
        let make = || QamDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
        check_block(&want, &input, SEED, make, &format!("qam_demod case {i}"));
        let mut block = make();
        let first = run_block(&mut block, &input, None);
        block.reset();
        same_bits(
            &first,
            &run_block(&mut block, &input, None),
            "qam_demod reset",
        );
        assert_eq!(block.equalizer().is_some(), taps != 0);
    }
}

#[test]
fn qam_demod_keeps_soft_bits_that_do_not_fit() {
    let fs = rate(38_400);
    let cfg = QamDemodConfig {
        order: 64,
        symbol_rate_hz: 9600.0,
        ..QamDemodConfig::default()
    };
    let input = received(64, 4, 2000, 0.01, 30);
    let want = demod_kernels(&cfg, fs, &input);
    let mut block = QamDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
    for room_len in [1usize, 5, 7] {
        block.reset();
        let mut got = Vec::new();
        let mut rest: &[C32] = &input;
        let mut room = vec![0.0f32; room_len];
        loop {
            let p = block.process(rest, &mut room).unwrap();
            assert!(p.produced <= room_len);
            got.extend_from_slice(&room[..p.produced]);
            rest = &rest[p.consumed..];
            if p.consumed == 0 && p.produced == 0 {
                break;
            }
        }
        assert_eq!(rest.len(), 0);
        same_bits(&want, &got, &format!("room {room_len}"));
    }
    // no output room: nothing is consumed
    block.reset();
    let p = block.process(&input, &mut []).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
}

#[test]
fn qam_demod_timing_rates_and_parameters() {
    let cfg = QamDemodConfig {
        order: 16,
        symbol_rate_hz: 25_000.0,
        ..QamDemodConfig::default()
    };
    let fs = rate(200_000);
    let block = QamDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
    assert_eq!(block.samples_per_symbol(), 8);
    assert_eq!(block.num_taps(), 81);
    let t = block.timing();
    assert_eq!(t.ratio, Ratio::new(1, 2).unwrap()); // 4 bits per 8 samples
    assert_eq!(t.kind, RatioKind::Variable);
    // 40 of the filter, 1 of the timing loop, 5 symbols of the equalizer
    assert_eq!(t.delay, Delay::new(80 + 2 + 80, 2).unwrap());
    assert_eq!(
        QamDemodBlock::<VecStorage>::rate_out(&cfg, fs).unwrap(),
        rate(100_000)
    );
    let no_eq = QamDemodConfig {
        equalizer_taps: 0,
        ..cfg
    };
    let b0 = QamDemodBlock::<VecStorage>::new(&no_eq, fs, &mut VecStorage).unwrap();
    assert_eq!(b0.timing().delay, Delay::new(82, 2).unwrap());
    assert!(b0.equalizer().is_none());
    let c64 = QamDemodConfig { order: 64, ..cfg };
    assert_eq!(
        QamDemodBlock::<VecStorage>::rate_out(&c64, fs).unwrap(),
        rate(150_000)
    );
    // the rate rule: 7.5 samples per symbol and below 2 are rejected
    for bad in [rate(187_500), rate(40_000)] {
        assert!(matches!(
            cfg.validate(bad),
            Err(Error::InvalidParameter {
                name: "symbol_rate_hz",
                ..
            })
        ));
    }
    // equalizer taps 0 or odd from 3; orders 16 and 64
    let mut c = QamDemodConfig::default();
    for (taps, ok) in [(0, true), (1, false), (3, true), (12, false), (63, true)] {
        assert_eq!(
            c.set("equalizer_taps", ParamValue::I64(taps)).is_ok(),
            ok,
            "{taps}"
        );
    }
    assert!(c.set("equalizer_taps", ParamValue::I64(65)).is_err());
    assert!(matches!(
        c.set("order", ParamValue::I64(20)),
        Err(Error::InvalidParameter { name: "order", .. })
    ));
    for (name, value) in [
        ("order", ParamValue::I64(64)),
        ("symbol_rate_hz", ParamValue::F64(1000.0)),
        ("rolloff", ParamValue::F64(0.25)),
        ("span", ParamValue::I64(12)),
        ("sync_bw", ParamValue::F64(0.02)),
        ("carrier_bw", ParamValue::F64(0.01)),
    ] {
        c.set(name, value).unwrap();
        assert_eq!(c.get(name), Ok(value), "{name}");
    }
    let d = sdrk_blocks::all_descriptions()
        .iter()
        .find(|d| d.kind == "qam_demod")
        .unwrap();
    assert_eq!(d.inputs[0].types, [PortType::C32]);
    assert_eq!(d.outputs[0].types, [PortType::F32]);
    assert_eq!(d.required, ["symbol_rate_hz"]);
    let names: Vec<_> = d.params.iter().map(|p| p.name).collect();
    assert_eq!(
        names,
        [
            "order",
            "symbol_rate_hz",
            "rolloff",
            "span",
            "equalizer_taps",
            "sync_bw",
            "carrier_bw"
        ]
    );
    for word in ["symbol_sync", "equalize", "costas", "rotations four"] {
        assert!(d.details.contains(word), "qam_demod details: {word}");
    }
}

/// The best alignment of the soft bits `soft` to the sent bits under the four rotations: the
/// rotation, the lag in symbols and the bit errors over the last `n` symbols.
fn best_match(sent: &[f32], soft: &[f32], k: usize, n: usize) -> (u32, usize, usize) {
    let sym = |b: &[f32]| b.iter().fold(0u32, |g, &x| (g << 1) | u32::from(x > 0.0));
    let tx: Vec<u32> = sent.chunks_exact(k).map(sym).collect();
    let rx: Vec<u32> = soft.chunks_exact(k).map(sym).collect();
    let mut best = (0, 0, usize::MAX);
    for r in 0..4 {
        for lag in 0..40.min(rx.len()) {
            let end = rx.len().min(tx.len() + lag);
            let errs = (end - n..end)
                .map(|i| {
                    let back = rotate_symbol(rx[i], (k / 2) as u32, (4 - r) % 4);
                    (back ^ tx[i - lag]).count_ones() as usize
                })
                .sum();
            if errs < best.2 {
                best = (r, lag, errs);
            }
        }
    }
    best
}

#[test]
fn qam_mod_to_qam_demod_with_little_noise_is_error_free() {
    // (order, samples per symbol, carrier offset in cycles per symbol): the frequency
    // acquisition brings 64-QAM the offsets of 16-QAM (before it, 64-QAM pulled in 0.0001 but
    // not 0.0005 here, I44-3) and both orders offsets up to the edge of its range of +-0.125
    for (i, (order, sps, offset)) in [
        (16u32, 4usize, 0.0005),
        (64, 4, 0.0005),
        (16, 8, 0.0005),
        (64, 4, 0.04),
        (16, 4, -0.1),
        (64, 8, -0.12),
    ]
    .into_iter()
    .enumerate()
    {
        let mcfg = QamModConfig {
            order,
            samples_per_symbol: sps,
            ..QamModConfig::default()
        };
        let k = mcfg.bits_per_symbol();
        let sent = random_bits(6000 * k, 40 + i as u64);
        let mut tx = QamModBlock::<VecStorage>::new(&mcfg, rate(48_000), &mut VecStorage).unwrap();
        let sig = run_block(&mut tx, &sent, None);
        let mut rng = Pcg32::new(SEED, 140 + i as u64);
        let noisy: Vec<C32> = sig
            .iter()
            .enumerate()
            .map(|(j, x)| {
                let ph = 1.1 + core::f64::consts::TAU * offset * j as f64 / sps as f64;
                let (s, c) = ph.sin_cos();
                let (re, im) = (f64::from(x.re), f64::from(x.im));
                C32::new(
                    (re * c - im * s + 0.005 * rng.normal()) as f32,
                    (re * s + im * c + 0.005 * rng.normal()) as f32,
                )
            })
            .collect();
        let dcfg = QamDemodConfig {
            order,
            symbol_rate_hz: 1000.0,
            ..QamDemodConfig::default()
        };
        let mut rx =
            QamDemodBlock::<VecStorage>::new(&dcfg, rate(1000 * sps as u64), &mut VecStorage)
                .unwrap();
        let soft = run_block(&mut rx, &noisy, None);
        let (r, lag, errs) = best_match(&sent, &soft, k, 2000);
        assert_eq!(errs, 0, "case {i}: rotation {r}, lag {lag}");
        assert!(rx.locked(), "case {i}");
        // the acquisition finds the offset, and the equalizer's phase tracker and the carrier
        // loop together hold it
        let a = rx.acquired_frequency().unwrap();
        assert!((a - offset).abs() < 1e-4, "case {i}: acquired {a}");
        let f = rx.frequency();
        assert!((f - offset).abs() < 1e-4, "case {i}: {f}");
    }
}

#[test]
fn qam_demod_acquires_once_after_its_window_and_again_after_reset() {
    // the window of the definition above is that of the block
    assert_eq!(
        (QAM_ACQUISITION_SKIP, QAM_ACQUISITION_SYMBOLS),
        (ACQ_SKIP, ACQ_SYMBOLS)
    );
    // 16-QAM at 4 samples per symbol with the offset 0.001 of `received`
    let fs = rate(38_400);
    for taps in [11usize, 0] {
        let cfg = QamDemodConfig {
            order: 16,
            symbol_rate_hz: 9600.0,
            equalizer_taps: taps,
            ..QamDemodConfig::default()
        };
        let input = received(16, 4, 2000, 0.01, 50);
        let mut block = QamDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
        let mut soft = vec![0.0f32; 4 * 4000];
        // 700 symbols: the window (symbols 256 to 767 of the timing loop) is not complete
        let p = block.process(&input[..2800], &mut soft).unwrap();
        assert_eq!(p.consumed, 2800);
        assert_eq!(block.acquired_frequency(), None, "taps {taps}");
        block.process(&input[2800..], &mut soft).unwrap();
        let f = block.acquired_frequency().unwrap();
        assert!((f - 0.001).abs() < 1e-4, "taps {taps}: {f}");
        match block.equalizer() {
            // the tracker took the estimate (and kept tracking)
            Some(eq) => assert!((f64::from(eq.frequency()) - 0.001).abs() < 1e-4),
            None => assert!((block.frequency() - 0.001).abs() < 1e-4),
        }
        block.reset();
        assert_eq!(block.acquired_frequency(), None);
        assert_eq!(block.frequency(), 0.0);
        // silence: no clear spectral peak, so no estimate is used
        let silence = vec![C32::default(); 4000];
        block.process(&silence, &mut soft).unwrap();
        assert_eq!(block.acquired_frequency(), None, "taps {taps}");
        assert_eq!(block.frequency(), 0.0, "taps {taps}");
        // noise of unit power for the first window, then the signal: a window without a clear
        // peak is followed by the next one, which finds the offset
        block.reset();
        let mut rng = Pcg32::new(SEED, 60);
        let mut late: Vec<C32> = (0..4 * 800)
            .map(|_| C32::new(rng.normal() as f32, rng.normal() as f32) * 0.7)
            .collect();
        late.extend_from_slice(&input);
        let p = block.process(&late[..4 * 790], &mut soft).unwrap();
        assert_eq!(p.consumed, 4 * 790);
        assert_eq!(block.acquired_frequency(), None, "taps {taps}");
        let mut rest = &late[4 * 790..];
        while !rest.is_empty() {
            rest = &rest[block.process(rest, &mut soft).unwrap().consumed..];
        }
        let f = block.acquired_frequency().unwrap();
        assert!((f - 0.001).abs() < 1e-4, "taps {taps}: late {f}");
    }
}

/// Runs `input` through `block` in calls of 4096 samples; the soft bits.
fn demod_all(block: &mut QamDemodBlock<VecStorage>, input: &[C32]) -> Vec<f32> {
    let mut soft = Vec::new();
    let mut out = vec![0.0f32; 8192];
    for piece in input.chunks(4096) {
        let mut rest = piece;
        while !rest.is_empty() {
            let p = block.process(rest, &mut out).unwrap();
            soft.extend_from_slice(&out[..p.produced]);
            rest = &rest[p.consumed..];
        }
    }
    soft
}

/// The noise per component of [`received`] for `Es/N0` in dB (the power of the signal measured
/// on its noiseless form after the ramp of the filter).
fn sigma_for(order: u32, sps: usize, n_sym: usize, stream: u64, esn0_db: f64) -> f64 {
    let clean = received(order, sps, n_sym, 0.0, stream);
    let body = &clean[100 * sps..];
    let power = body
        .iter()
        .map(|c| f64::from(c.re).powi(2) + f64::from(c.im).powi(2))
        .sum::<f64>()
        / body.len() as f64;
    (power * sps as f64 / (2.0 * 10f64.powf(esn0_db / 10.0))).sqrt()
}

#[test]
fn qam_demod_acquisition_phases_are_independent_of_the_cut_plan() {
    // F5 of the review of I44-9 (D1435): the phases of the acquisition (the first interval, the
    // checks without lock, the verify window, the sampled intervals and the restart) act at the
    // same symbols whatever the split into calls. A weak 16-QAM signal below the lock level
    // (Es/N0 12 dB, three checks without lock: verify, sampled intervals) from the start, and
    // the same signal after a carrier at 0.06 cycles per symbol (its estimate, then the restart
    // by the verify window); whole, in the twenty cut plans (one-sample pieces among them) and
    // after a reset.
    let fs = rate(38_400);
    let n_sig = QAM_ACQUISITION_SKIP + QAM_ACQUISITION_SYMBOLS + 3 * QAM_ACQUISITION_CHECK + 200;
    let sigma = sigma_for(16, 4, n_sig, 90, 12.0);
    let sig = received(16, 4, n_sig, sigma, 90);
    let amp = 0.7f64;
    let mut lead: Vec<C32> = (0..4 * 1000)
        .map(|j| {
            let ph = core::f64::consts::TAU * 0.06 * j as f64 / 4.0;
            C32::new((amp * ph.cos()) as f32, (amp * ph.sin()) as f32)
        })
        .collect();
    lead.extend_from_slice(&sig);
    for (name, input, restarted) in [("weak", &sig, false), ("carrier, weak", &lead, true)] {
        for taps in [11usize, 0] {
            let cfg = QamDemodConfig {
                order: 16,
                symbol_rate_hz: 9600.0,
                equalizer_taps: taps,
                ..QamDemodConfig::default()
            };
            let what = format!("{name}, taps {taps}");
            let make = || QamDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
            let mut block = make();
            let want = run_block(&mut block, input, None);
            let state = (block.restarts(), block.acquired_frequency(), block.locked());
            assert!(!state.2, "{what}: the weak signal is below the lock level");
            assert_eq!(state.0 > 0, restarted, "{what}");
            let f = state.1.unwrap();
            assert!((f - 0.001).abs() < 1e-4, "{what}: acquired {f}");
            for plan in twenty_plans(SEED) {
                let mut block = make();
                let got = run_block(&mut block, input, Some(plan));
                same_bits(&want, &got, &format!("{what} {plan:?}"));
                assert_eq!(
                    (block.restarts(), block.acquired_frequency(), block.locked()),
                    state,
                    "{what} {plan:?}"
                );
            }
            block.reset();
            same_bits(
                &want,
                &run_block(&mut block, input, None),
                &format!("{what} reset"),
            );
            assert_eq!(block.restarts(), state.0, "{what} reset");
        }
    }
}

#[test]
fn qam_demod_lock_and_restart_on_the_level_are_independent_of_the_cut_plan() {
    // N1 of the review of I44-9 and I44-11: the phases that the test above does not reach act
    // at the same symbols whatever the split into calls. A strong 16-QAM signal for 6000
    // symbols, then noise: the estimate, lock at the first check (the interval after it takes
    // no samples), the lock lost in that interval and so a check without lock and without
    // samples (the verify window, on noise no clear peak), then the 16 samples of noise and the
    // restart on their mean. And a carrier at 0.06 cycles per symbol, then noise: its estimate,
    // the first check without lock (in the noise), the verify window and the restart on the
    // mean. Whole, in the twenty cut plans (one-sample pieces among them) and after a reset.
    let fs = rate(38_400);
    let mut rng = Pcg32::new(SEED, 92);
    let noise: Vec<C32> = (0..4 * 9000)
        .map(|_| C32::new(rng.normal() as f32, rng.normal() as f32) * 0.7)
        .collect();
    let strong = received(16, 4, 6000, 0.01, 91);
    let mut strong_noise = strong.clone();
    strong_noise.extend_from_slice(&noise);
    let mut carrier_noise: Vec<C32> = (0..4 * 1000)
        .map(|j| {
            let ph = core::f64::consts::TAU * 0.06 * j as f64 / 4.0;
            C32::new((0.7 * ph.cos()) as f32, (0.7 * ph.sin()) as f32)
        })
        .collect();
    carrier_noise.extend_from_slice(&noise);
    // the checks at about symbol 768 + 4096 k; `before` some 300 symbols before the one that
    // restarts (the third for the strong signal, the second for the carrier)
    for (name, input, lead, before, f_want) in [
        (
            "strong, noise",
            &strong_noise,
            strong.len(),
            4 * 12_750,
            0.001,
        ),
        ("carrier, noise", &carrier_noise, 4 * 1000, 4 * 8650, 0.06),
    ] {
        for taps in [11usize, 0] {
            let cfg = QamDemodConfig {
                order: 16,
                symbol_rate_hz: 9600.0,
                equalizer_taps: taps,
                ..QamDemodConfig::default()
            };
            let what = format!("{name}, taps {taps}");
            let make = || QamDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
            // the path: the estimate (and lock on the strong signal) in the lead, then exactly
            // one restart
            let mut block = make();
            run_block(&mut block, &input[..lead], None);
            let f = block.acquired_frequency().unwrap();
            assert!(
                (f - f_want).abs() < 1e-4,
                "{what}: acquired {f} in the lead"
            );
            if lead == strong.len() {
                assert!(block.locked(), "{what}: lock after the lead");
            }
            assert_eq!(block.restarts(), 0, "{what}: after the lead");
            // no restart before the check that takes the mean (the verify window kept it)
            run_block(&mut block, &input[lead..before], None);
            assert_eq!(block.restarts(), 0, "{what}: before the check on the mean");
            let mut block = make();
            let want = run_block(&mut block, input, None);
            let state = (block.restarts(), block.acquired_frequency(), block.locked());
            assert_eq!(state.0, 1, "{what}: one restart, on the mean of the noise");
            assert!(!state.2, "{what}");
            let f = state.1.unwrap();
            assert!((f - f_want).abs() < 1e-4, "{what}: acquired {f}");
            for plan in twenty_plans(SEED) {
                let mut block = make();
                let got = run_block(&mut block, input, Some(plan));
                same_bits(&want, &got, &format!("{what} {plan:?}"));
                assert_eq!(
                    (block.restarts(), block.acquired_frequency(), block.locked()),
                    state,
                    "{what} {plan:?}"
                );
            }
            block.reset();
            same_bits(
                &want,
                &run_block(&mut block, input, None),
                &format!("{what} reset"),
            );
            assert_eq!(block.restarts(), state.0, "{what} reset");
        }
    }
}

#[test]
fn qam_demod_searches_again_when_the_loop_does_not_lock() {
    // R1 of the review of the QAM round (D1421): an estimate taken before the signal (here
    // from a carrier at 0.06 cycles per symbol, whose fourth power is a clear peak, and from
    // Gaussian noise for 40 windows) is held only while the carrier loop locks; at the checks
    // every `QAM_ACQUISITION_CHECK` symbols without lock the acquisition searches again and finds
    // the signal (16-QAM at 4 samples per symbol, offset 0.001 cycles per symbol, the block
    // defaults with `carrier_bw` 0.005).
    assert_eq!(QAM_ACQUISITION_CHECK, 4096);
    let fs = rate(38_400);
    let n_sig = 12_000;
    let mut rng = Pcg32::new(SEED, 70);
    let tone: Vec<C32> = (0..4 * 6000)
        .map(|j| {
            let ph = core::f64::consts::TAU * 0.06 * j as f64 / 4.0;
            C32::new((0.7 * ph.cos()) as f32, (0.7 * ph.sin()) as f32)
        })
        .collect();
    let noise: Vec<C32> = (0..4 * 40 * QAM_ACQUISITION_SYMBOLS)
        .map(|_| C32::new(rng.normal() as f32, rng.normal() as f32) * 0.7)
        .collect();
    for (lead_name, lead) in [("carrier", &tone), ("noise", &noise)] {
        for taps in [11usize, 0] {
            let cfg = QamDemodConfig {
                order: 16,
                symbol_rate_hz: 9600.0,
                equalizer_taps: taps,
                ..QamDemodConfig::default()
            };
            let mut input = lead.clone();
            input.extend_from_slice(&received(16, 4, n_sig, 0.01, 71));
            let mut block = QamDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
            let soft = demod_all(&mut block, &input);
            let what = format!("{lead_name}, taps {taps}");
            let f = block.acquired_frequency().unwrap();
            assert!((f - 0.001).abs() < 1e-4, "{what}: acquired {f}");
            assert!(block.locked(), "{what}");
            // the last 2000 symbols against the sent ones at the best rotation and offset (the
            // timing loop gives about one symbol per 4 samples of the lead-in, not exactly)
            let sym = |b: &[f32]| b.iter().fold(0u32, |g, &x| (g << 1) | u32::from(x > 0.0));
            let tx: Vec<u32> = random_bits(n_sig * 4, 71)
                .chunks_exact(4)
                .map(sym)
                .collect();
            let rx: Vec<u32> = soft.chunks_exact(4).map(sym).collect();
            let lead_syms = lead.len() / 4;
            let errs = (0..4u32)
                .flat_map(|r| (lead_syms - 300..lead_syms + 300).map(move |o| (r, o)))
                .map(|(r, o)| {
                    (rx.len() - 2000..rx.len())
                        .map(|i| {
                            let back = rotate_symbol(rx[i], 2, (4 - r) % 4);
                            tx.get(i.wrapping_sub(o))
                                .map_or(4, |t| (back ^ t).count_ones())
                        })
                        .sum::<u32>()
                })
                .min()
                .unwrap();
            assert_eq!(errs, 0, "{what}");
        }
    }
}

// ---- frame_decode rotations four (M44 §4, test 1) ----------------------------------------

/// The frame of `data` (4 preamble bytes) after `lead` random bits, as bits `0`/`1`, padded with
/// random bits to whole symbols of `k` bits and 4 symbols more.
fn frame_stream(data: &[u8], lead: usize, k: usize, stream: u64) -> Vec<u8> {
    let p = Packet::new(0, PacketFlags::NONE, data).unwrap();
    let mut frame = vec![0u8; frame_bits(data.len(), 4)];
    encode(&p, 4, Whitening::None, &mut frame).unwrap();
    let mut rng = Pcg32::new(SEED, stream);
    let mut bits: Vec<u8> = (0..lead).map(|_| (rng.next_u32() & 1) as u8).collect();
    bits.extend_from_slice(&frame);
    while !bits.len().is_multiple_of(k) {
        bits.push((rng.next_u32() & 1) as u8);
    }
    bits.extend((0..4 * k).map(|_| (rng.next_u32() & 1) as u8));
    bits
}

/// Every symbol of `k` bits of `bits` turned by `r` quarter turns with the rotation map of
/// `sdrk_core`, as soft bits `+-1`.
fn turned_soft(bits: &[u8], k: usize, r: u32) -> Vec<f32> {
    bits.chunks_exact(k)
        .flat_map(|c| {
            let g = c.iter().fold(0u32, |g, &b| (g << 1) | u32::from(b));
            let t = rotate_symbol(g, (k / 2) as u32, r);
            (0..k)
                .rev()
                .map(move |j| if (t >> j) & 1 == 1 { 1.0 } else { -1.0 })
        })
        .collect()
}

fn decoder(rotations: FrameRotations, k: usize) -> FrameDecodeBlock<VecStorage> {
    let cfg = FrameDecodeConfig {
        rotations,
        bits_per_symbol: k,
        ..FrameDecodeConfig::default()
    };
    FrameDecodeBlock::<VecStorage>::new(&cfg, rate(9600), &mut VecStorage).unwrap()
}

/// The packets of `soft` through `dec`, in pieces of the plan (`None`: whole).
fn decode_all(
    dec: &mut FrameDecodeBlock<VecStorage>,
    soft: &[f32],
    plan: Option<sdrk_testkit::ChunkPlan>,
) -> Vec<Packet> {
    let mut got = Vec::new();
    let mut out = vec![Packet::default(); 2];
    let mut pos = 0;
    let mut sizes = plan.map(|p| p.sizes());
    while pos < soft.len() {
        let want = sizes
            .as_mut()
            .map_or(soft.len(), |s| s.next().unwrap())
            .min(soft.len() - pos);
        let mut piece = &soft[pos..pos + want];
        pos += want;
        while !piece.is_empty() {
            let p = dec.process(piece, &mut out).unwrap();
            got.extend_from_slice(&out[..p.produced]);
            piece = &piece[p.consumed..];
        }
    }
    got
}

#[test]
fn proto_rotation_map_equals_the_constellation_map() {
    for bpa in 1..=3u32 {
        for s in 0..(1u32 << (2 * bpa)) {
            for r in 0..6 {
                assert_eq!(
                    rotate_group(s, bpa, r),
                    rotate_symbol(s, bpa, r),
                    "{bpa} {s} {r}"
                );
            }
        }
    }
}

#[test]
fn frame_decode_four_finds_every_rotation_for_2_4_and_6_bits_per_symbol() {
    let data = b"rotations of the sync word";
    for k in [2usize, 4, 6] {
        for lead in [0usize, 1, k - 1, 7] {
            let bits = frame_stream(data, lead, k, (k * 10 + lead) as u64);
            for r in 0..4u32 {
                let soft = turned_soft(&bits, k, r);
                let what = format!("k {k}, lead {lead}, rotation {r}");
                let got = decode_all(&mut decoder(FrameRotations::Four, k), &soft, None);
                assert_eq!(got.len(), 1, "{what}");
                assert_eq!(got[0].data(), data, "{what}");
                assert!(got[0].flags.contains(PacketFlags::CRC_OK), "{what}");
                assert_eq!(rotation_of(got[0].flags), r, "{what}");
                assert_eq!(
                    got[0].flags.bits(),
                    PacketFlags::CRC_OK.bits() | (r << ROTATION_SHIFT),
                    "{what}"
                );
                assert_eq!(got[0].index, (lead + 32) as u64, "{what}");
                // normal finds only the unturned frame and reports no rotation
                let plain = decode_all(&mut decoder(FrameRotations::Normal, k), &soft, None);
                if r == 0 {
                    assert_eq!(plain.len(), 1, "{what}");
                    assert_eq!(plain[0].flags, PacketFlags::CRC_OK, "{what}");
                } else {
                    assert!(plain.is_empty(), "{what}");
                }
                // both finds the half turn exactly for QPSK (the inversion of both bits)
                if k == 2 {
                    let both = decode_all(&mut decoder(FrameRotations::Both, k), &soft, None);
                    assert_eq!(both.len(), usize::from(r % 2 == 0), "{what}");
                }
            }
        }
    }
}

#[test]
fn frame_decode_four_finds_frames_of_different_rotations_in_cut_plans() {
    for k in [2usize, 4, 6] {
        let mut soft = Vec::new();
        let mut want = Vec::new();
        for (j, r) in [1u32, 0, 3, 2, 2].into_iter().enumerate() {
            let data: Vec<u8> = (0..(5 + 7 * j) as u8).collect();
            let bits = frame_stream(&data, j, k, (100 + k * 10 + j) as u64);
            let start = soft.len();
            soft.extend(turned_soft(&bits, k, r));
            want.push(((start + j + 32) as u64, data, r));
        }
        let check = |got: &[Packet], what: &str| {
            assert_eq!(got.len(), want.len(), "{what}");
            for (g, (index, data, r)) in got.iter().zip(&want) {
                assert_eq!(
                    (g.index, g.data(), rotation_of(g.flags)),
                    (*index, &data[..], *r),
                    "{what}"
                );
                assert!(g.flags.contains(PacketFlags::CRC_OK), "{what}");
            }
        };
        let mut dec = decoder(FrameRotations::Four, k);
        let whole = decode_all(&mut dec, &soft, None);
        check(&whole, &format!("k {k} whole"));
        assert_eq!(dec.position(), soft.len() as u64);
        for plan in twenty_plans(SEED) {
            check(
                &decode_all(&mut decoder(FrameRotations::Four, k), &soft, Some(plan)),
                &format!("k {k} {plan:?}"),
            );
        }
        // reset starts the symbol count over
        dec.reset();
        check(
            &decode_all(&mut dec, &soft, None),
            &format!("k {k} after reset"),
        );
    }
}

#[test]
fn frame_decode_rotations_and_bits_per_symbol_are_checked() {
    let make = |rotations, bits_per_symbol| {
        let cfg = FrameDecodeConfig {
            rotations,
            bits_per_symbol,
            ..FrameDecodeConfig::default()
        };
        FrameDecodeBlock::<VecStorage>::new(&cfg, rate(9600), &mut VecStorage).map(|_| ())
    };
    for k in [2, 4, 6] {
        assert!(make(FrameRotations::Four, k).is_ok(), "{k}");
    }
    for k in [1, 3, 5] {
        assert!(matches!(
            make(FrameRotations::Four, k),
            Err(Error::InvalidParameter {
                name: "bits_per_symbol",
                ..
            })
        ));
        assert!(make(FrameRotations::Normal, k).is_ok());
        assert!(make(FrameRotations::Both, k).is_ok());
    }
    for k in [0, 7] {
        assert!(matches!(
            make(FrameRotations::Normal, k),
            Err(Error::OutOfRange {
                name: "bits_per_symbol",
                ..
            })
        ));
    }
}

/// Packets through `frame_encode` -> `qam_mod` -> a channel turned by 0, 90, 180 and 270
/// degrees -> `qam_demod` -> `frame_decode` `four`: every channel rotation delivers the packets,
/// and the reported rotations step with the channel (the receiver is equivariant to a rotation
/// of its input by 90 degrees up to rounding).
#[test]
fn qam_link_with_frame_decode_four_resolves_every_rotation() {
    for (order, k) in [(16u32, 4usize), (64, 6)] {
        let mut rng = Pcg32::new(SEED, 200 + u64::from(order));
        let packets: Vec<Packet> = (0..12)
            .map(|_| {
                let len = 1 + rng.below(40) as usize;
                let data: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
                Packet::new(0, PacketFlags::NONE, &data).unwrap()
            })
            .collect();
        // 4000 symbols of random bits for the training (the preamble 0x55 is one constant
        // symbol for 16-QAM, on which neither the timing loop nor the blind equalizer can
        // train), then the frames (4 preamble bytes) with 50 random bits between them
        let random = |rng: &mut Pcg32, n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
                .collect()
        };
        let mut bits = random(&mut rng, 4000 * k);
        for p in &packets {
            let mut frame = vec![0u8; frame_bits(p.data().len(), 4)];
            encode(p, 4, Whitening::None, &mut frame).unwrap();
            bits.extend(frame.iter().map(|&b| if b == 1 { 1.0f32 } else { -1.0 }));
            bits.extend(random(&mut rng, 50));
        }
        bits.extend(random(&mut rng, 200 * k));
        bits.truncate(bits.len() / k * k);
        let mcfg = QamModConfig {
            order,
            samples_per_symbol: 4,
            ..QamModConfig::default()
        };
        let mut tx = QamModBlock::<VecStorage>::new(&mcfg, rate(4000), &mut VecStorage).unwrap();
        let sig = run_block(&mut tx, &bits, None);
        let mut rots = Vec::new();
        for turn in 0..4u32 {
            let turned: Vec<C32> = sig
                .iter()
                .map(|x| match turn {
                    0 => *x,
                    1 => C32::new(-x.im, x.re),
                    2 => C32::new(-x.re, -x.im),
                    _ => C32::new(x.im, -x.re),
                })
                .collect();
            let dcfg = QamDemodConfig {
                order,
                symbol_rate_hz: 1000.0,
                ..QamDemodConfig::default()
            };
            let mut rx =
                QamDemodBlock::<VecStorage>::new(&dcfg, rate(4000), &mut VecStorage).unwrap();
            let soft = run_block(&mut rx, &turned, None);
            let got = decode_all(&mut decoder(FrameRotations::Four, k), &soft, None);
            let what = format!("{order}-QAM turned {turn}");
            assert_eq!(got.len(), packets.len(), "{what}");
            for (g, s) in got.iter().zip(&packets) {
                assert_eq!(g.data(), s.data(), "{what}");
                assert!(g.flags.contains(PacketFlags::CRC_OK), "{what}");
            }
            let r = rotation_of(got[0].flags);
            assert!(got.iter().all(|g| rotation_of(g.flags) == r), "{what}");
            rots.push(r);
        }
        for turn in 1..4 {
            assert_eq!(
                rots[turn],
                (rots[0] + turn as u32) % 4,
                "{order}-QAM {rots:?}"
            );
        }
    }
}
