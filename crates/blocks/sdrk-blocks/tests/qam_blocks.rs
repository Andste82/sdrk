//! The blocks of M44: the `slice` constellations `qam16` and `qam64` (test 1: the soft-bit
//! signs of every point derived from the mapping of `sdrk_core::kernels::constellation`, the
//! piecewise-linear max-log values bit for bit, the agreement of every sign with the grid
//! decision, the rates, the parameter checks, the extremes and 20 cut plans of seed 607), and the
//! blocks `costas` and `equalize` with the constellations `qam16` and `qam64` (M44 §3: the block
//! equals the kernel whole and in 20 cut plans of seed 607, the final state, `reset`, lock and
//! frequency; the constellation choice).
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

use common::{check_block, rate, run_block, same_bits};
use sdrk::{C32, Delay, Ratio, RatioKind, Sdrk};
use sdrk_blocks::{
    Block, CostasBlock, CostasConfig, CostasConstellation, EqConstellation, EqMode, EqualizeBlock,
    EqualizeConfig, ParamValue, PortType, SliceBlock, SliceConfig, SliceConstellation,
};
use sdrk_core::kernels::constellation::Qam;
use sdrk_core::kernels::costas::{Costas, CostasState};
use sdrk_core::kernels::equalize::{EqParams, Equalize, EqualizeState};
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

/// `a` and `b` bit for bit (NaN included).
fn same_f32_bits(a: &[f32], b: &[f32], what: &str) {
    let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
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
