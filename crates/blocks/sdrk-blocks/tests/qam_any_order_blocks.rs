//! The building blocks of M44.2 §3 with the new square orders: `costas` and `equalize` with
//! `qam256` and `qam1024` equal their kernels whole and in 20 cut plans, with the choice indices
//! of `symbol_map` (M44.1.1); `symbol_map` -> a quarter turn -> `symbol_demap` -> `frame_decode`
//! `four` delivers the packets with the turn for `qpsk` and every QAM order (the symbol groups of
//! up to 10 bits, §4); `symbol_map` and `symbol_demap` agree on every point of `qam256` and
//! `qam1024`. Seed 839 (M44.2). Test 8 of M44.2 (seed 837): `qam_mod` and `qam_demod` with
//! `qpsk`, `qam256` and `qam1024` do not depend on the cut plan.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::approx_constant,
    clippy::too_many_lines
)]

mod common;

use common::{check_block, rate, run_block, same_bits};
use sdrk::{C32, Sdrk};
use sdrk_blocks::{
    Block, CostasBlock, CostasConfig, CostasConstellation, EqConstellation, EqualizeBlock,
    EqualizeConfig, FrameDecodeBlock, FrameDecodeConfig, FrameRotations, Packet, PacketFlags,
    ParamValue, QamDemodBlock, QamDemodConfig, QamModBlock, QamModConfig, SymbolDemapBlock,
    SymbolDemapConfig, SymbolDemapConstellation, SymbolMapBlock, SymbolMapConfig,
    SymbolMapConstellation, qam_acquisition_check,
};
use sdrk_core::kernels::constellation::Qam;
use sdrk_core::kernels::costas::{Costas, CostasState};
use sdrk_core::kernels::equalize::{EqParams, Equalize, EqualizeState};
use sdrk_proto::frame::{Whitening, encode, frame_bits, rotation_of};
use sdrk_testkit::{Pcg32, VecStorage};

/// The seed of M44.2 for these tests.
const SEED: u64 = 839;

/// The new orders: the constellation of `symbol_map`, `costas`, `equalize` and `symbol_demap`,
/// and the choice index (the same in every block).
const NEW: [(
    Qam,
    SymbolMapConstellation,
    CostasConstellation,
    EqConstellation,
    SymbolDemapConstellation,
    u16,
); 2] = [
    (
        Qam::Qam256,
        SymbolMapConstellation::Qam256,
        CostasConstellation::Qam256,
        EqConstellation::Qam256,
        SymbolDemapConstellation::Qam256,
        5,
    ),
    (
        Qam::Qam1024,
        SymbolMapConstellation::Qam1024,
        CostasConstellation::Qam1024,
        EqConstellation::Qam1024,
        SymbolDemapConstellation::Qam1024,
        6,
    ),
];

/// Random points of `q`, turned by `phase0 + 2 pi offset k`, plus Gaussian noise of `sigma` per
/// component.
fn qam_signal(q: Qam, n: usize, phase0: f32, offset: f32, sigma: f32, stream: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED, stream);
    (0..n)
        .map(|k| {
            let p = q.point(rng.next_u32());
            let (sn, cs) = (phase0 + core::f32::consts::TAU * offset * k as f32).sin_cos();
            C32::new(
                p.re * cs - p.im * sn + sigma * rng.normal() as f32,
                p.re * sn + p.im * cs + sigma * rng.normal() as f32,
            )
        })
        .collect()
}

#[test]
fn the_new_orders_are_choices_of_every_block() {
    for (q, m, c, e, d, i) in NEW {
        let mut map = SymbolMapConfig::default();
        map.set("constellation", ParamValue::Choice(i)).unwrap();
        assert_eq!(map.constellation, m);
        assert_eq!(map.bits_per_symbol(), q.bits_per_symbol());
        let mut costas = CostasConfig::default();
        costas.set("constellation", ParamValue::Choice(i)).unwrap();
        assert_eq!(costas.constellation, c);
        assert_eq!(costas.get("constellation"), Ok(ParamValue::Choice(i)));
        let mut eq = EqualizeConfig::default();
        eq.set("constellation", ParamValue::Choice(i)).unwrap();
        assert_eq!(eq.constellation, e);
        assert_eq!(eq.get("constellation"), Ok(ParamValue::Choice(i)));
        let mut demap = SymbolDemapConfig::default();
        demap.set("constellation", ParamValue::Choice(i)).unwrap();
        assert_eq!(demap.constellation, d);
        assert_eq!(d.bits_per_symbol(), q.bits_per_symbol());
        // The conversions of the composites name the same grid.
        assert_eq!(CostasConstellation::from(m), c);
        assert_eq!(EqConstellation::from(m), e);
        assert_eq!(SymbolDemapConstellation::from(m), d);
    }
}

#[test]
fn costas_block_with_the_new_orders_equals_the_kernel() {
    for (stream, (q, _, c, _, _, i)) in NEW.into_iter().enumerate() {
        let cfg = CostasConfig {
            constellation: c,
            loop_bw: 0.02,
            ..CostasConfig::default()
        };
        // Eb/N0 about 30 dB (256-QAM) and 34 dB (1024-QAM).
        let sigma = if q == Qam::Qam256 { 0.0056 } else { 0.0028 };
        let input = qam_signal(q, 6000, 0.1, 0.0002, sigma, stream as u64);
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
            (block.frequency() - 0.0002).abs() < 2e-4,
            "{q:?}: {}",
            block.frequency()
        );
        // The tail sits on the grid.
        for y in &got[4000..] {
            let d = q.decide(*y);
            let spacing = 2.0 / q.sqrt_energy();
            assert!(
                (y.re - d.re).hypot(y.im - d.im) < 0.4 * spacing,
                "{q:?} {y}"
            );
        }
        assert_eq!(block.param("constellation"), Ok(ParamValue::Choice(i)));
    }
}

#[test]
fn equalize_block_with_the_new_orders_equals_the_kernel() {
    for (stream, (q, _, _, e, _, i)) in NEW.into_iter().enumerate() {
        let cfg = EqualizeConfig {
            taps: 11,
            constellation: e,
            mu_cma: 0.02,
            mu_dd: 0.005,
            cma_symbols: 2000,
            phase_bw: 0.005,
            ..EqualizeConfig::default()
        };
        let clean = qam_signal(q, 5000, 0.2, 0.0, 0.002, 10 + stream as u64);
        let h1 = C32::new(0.15, -0.1);
        let input: Vec<C32> = (0..clean.len())
            .map(|k| {
                if k > 0 {
                    clean[k] + clean[k - 1] * h1
                } else {
                    clean[k]
                }
            })
            .collect();
        let p = EqParams {
            constellation: e,
            feedback_taps: 0,
            mu_cma: 0.02,
            mu_dd: 0.005,
            cma_symbols: 2000,
            phase_bw: 0.005,
        };
        let mut st =
            EqualizeState::new(vec![C32::default(); 11], vec![C32::default(); 11], p).unwrap();
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
        assert_eq!(block.param("constellation"), Ok(ParamValue::Choice(i)));
    }
}

/// `symbol_map` gives the points of `Qam`, and `symbol_demap` gives every point its own bits
/// back (the soft bits of the kernel `qam_soft_bits`, the signs of the mapping).
#[test]
fn symbol_map_and_symbol_demap_agree_on_every_point() {
    for (q, m, _, _, d, _) in NEW {
        let k = q.bits_per_symbol();
        let n = q.points();
        let soft: Vec<f32> = (0..n)
            .flat_map(|s| {
                (0..k).map(move |j| {
                    if (s >> (k - 1 - j)) & 1 == 1 {
                        1.0
                    } else {
                        -1.0
                    }
                })
            })
            .collect();
        let mut map = SymbolMapBlock::<VecStorage>::new(
            &SymbolMapConfig {
                constellation: m,
                ..SymbolMapConfig::default()
            },
            rate(9600),
            &mut VecStorage,
        )
        .unwrap();
        let points = run_block(&mut map, &soft, None);
        same_bits(&points, q.point_table(), &format!("{q:?} points"));
        let mut demap = SymbolDemapBlock::<C32, VecStorage>::new(
            &SymbolDemapConfig {
                constellation: d,
                scale: 1.0,
            },
            rate(9600),
            &mut VecStorage,
        )
        .unwrap();
        let back = run_block(&mut demap, &points, None);
        assert_eq!(back.len(), soft.len());
        for (j, (b, s)) in back.iter().zip(&soft).enumerate() {
            assert_eq!(*b > 0.0, *s > 0.0, "{q:?} soft bit {j}: {b}");
        }
    }
}

/// M44.2 §4: frames through `symbol_map`, every symbol turned by 0 to 3 quarter turns,
/// `symbol_demap` and `frame_decode` `four`: every turn delivers the packets for `qpsk` and the
/// QAM of 16 to 1024 points (symbol groups of 2 to 10 bits) and reports the turn.
#[test]
fn frame_decode_four_resolves_every_turn_of_every_square_order() {
    let all = [
        (
            Qam::Qpsk,
            SymbolMapConstellation::Qpsk,
            SymbolDemapConstellation::Qpsk,
        ),
        (
            Qam::Qam16,
            SymbolMapConstellation::Qam16,
            SymbolDemapConstellation::Qam16,
        ),
        (
            Qam::Qam64,
            SymbolMapConstellation::Qam64,
            SymbolDemapConstellation::Qam64,
        ),
        (
            Qam::Qam256,
            SymbolMapConstellation::Qam256,
            SymbolDemapConstellation::Qam256,
        ),
        (
            Qam::Qam1024,
            SymbolMapConstellation::Qam1024,
            SymbolDemapConstellation::Qam1024,
        ),
    ];
    for (q, m, d) in all {
        let k = q.bits_per_symbol();
        let mut rng = Pcg32::new(SEED, 100 + k as u64);
        let packets: Vec<Packet> = (0..4)
            .map(|_| {
                let len = 1 + rng.below(30) as usize;
                let data: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
                Packet::new(0, PacketFlags::NONE, &data).unwrap()
            })
            .collect();
        let mut bits: Vec<f32> = Vec::new();
        for p in &packets {
            let mut frame = vec![0u8; frame_bits(p.data().len(), 2)];
            encode(p, 2, Whitening::Pn9, &mut frame).unwrap();
            bits.extend(frame.iter().map(|&b| if b == 1 { 1.0f32 } else { -1.0 }));
            bits.extend((0..37).map(|_| {
                if rng.next_u32() & 1 == 1 {
                    1.0f32
                } else {
                    -1.0
                }
            }));
        }
        bits.extend(core::iter::repeat_n(-1.0f32, 3 * k));
        bits.truncate(bits.len() / k * k);
        let mut map = SymbolMapBlock::<VecStorage>::new(
            &SymbolMapConfig {
                constellation: m,
                ..SymbolMapConfig::default()
            },
            rate(9600),
            &mut VecStorage,
        )
        .unwrap();
        let sig = run_block(&mut map, &bits, None);
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
            let mut demap = SymbolDemapBlock::<C32, VecStorage>::new(
                &SymbolDemapConfig {
                    constellation: d,
                    scale: 1.0,
                },
                rate(9600),
                &mut VecStorage,
            )
            .unwrap();
            let soft = run_block(&mut demap, &turned, None);
            let cfg = FrameDecodeConfig {
                rotations: FrameRotations::Four,
                constellation: m,
                whitening: Whitening::Pn9,
                ..FrameDecodeConfig::default()
            };
            assert_eq!(cfg.bits_per_symbol(), k);
            let mut dec =
                FrameDecodeBlock::<VecStorage>::new(&cfg, rate(9600), &mut VecStorage).unwrap();
            let mut got = Vec::new();
            let mut out = vec![Packet::default(); 4];
            let mut piece = &soft[..];
            while !piece.is_empty() {
                let p = dec.process(piece, &mut out).unwrap();
                got.extend_from_slice(&out[..p.produced]);
                piece = &piece[p.consumed..];
            }
            let what = format!("{q:?} turned {turn}");
            assert_eq!(got.len(), packets.len(), "{what}");
            for (g, s) in got.iter().zip(&packets) {
                assert_eq!(g.data(), s.data(), "{what}");
                assert!(g.flags.contains(PacketFlags::CRC_OK), "{what}");
                assert_eq!(rotation_of(g.flags), turn, "{what}");
            }
        }
    }
}

/// The seed of test 8 of M44.2 (the chunk invariance of the QAM composites).
const SEED_CHUNKS: u64 = 837;

/// A `qam_mod` signal of `n` random symbols of `c` (4 samples per symbol), turned by
/// `0.6 + 2 pi 0.001 k / 4` at sample `k` (0.001 cycles per symbol) with Gaussian noise of
/// `sigma` per component (seed 837, stream `stream`); the soft bits sent and the samples.
fn composite_signal(
    c: SymbolMapConstellation,
    n: usize,
    sigma: f64,
    stream: u64,
) -> (Vec<f32>, Vec<C32>) {
    let cfg = QamModConfig {
        constellation: c,
        samples_per_symbol: 4,
        ..QamModConfig::default()
    };
    let k = cfg.bits_per_symbol();
    let mut rng = Pcg32::new(SEED_CHUNKS, stream);
    let bits: Vec<f32> = (0..n * k)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect();
    let mut tx = QamModBlock::<VecStorage>::new(&cfg, rate(40_000), &mut VecStorage).unwrap();
    let sig = run_block(&mut tx, &bits, None);
    let mut noise = Pcg32::new(SEED_CHUNKS, 100 + stream);
    let input = sig
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let ph = 0.6 + core::f64::consts::TAU * 0.001 * i as f64 / 4.0;
            let (s, co) = ph.sin_cos();
            let (re, im) = (f64::from(x.re), f64::from(x.im));
            C32::new(
                (re * co - im * s + sigma * noise.normal()) as f32,
                (re * s + im * co + sigma * noise.normal()) as f32,
            )
        })
        .collect();
    (bits, input)
}

/// M44.2 test 8 (seed 837): `qam_mod` and `qam_demod` with `qpsk`, `qam256` and `qam1024` give
/// the same output whole and in 20 cut plans. The receivers run past the first check of the
/// controller (`qam_acquisition_check` of the order after the estimate: the acquisition, the
/// switch of the equalizer to the decision mode with its hand-over, the first check), with and
/// without the equalizer.
#[test]
fn qam_mod_and_qam_demod_of_every_new_order_do_not_depend_on_the_cut_plan() {
    for (i, (c, sigma, taps)) in [
        (SymbolMapConstellation::Qpsk, 0.1, 11usize),
        (SymbolMapConstellation::Qam256, 0.01, 11),
        (SymbolMapConstellation::Qam1024, 0.005, 11),
        (SymbolMapConstellation::Qam256, 0.01, 0),
    ]
    .into_iter()
    .enumerate()
    {
        let n = qam_acquisition_check(c) + 2000;
        let (bits, input) = composite_signal(c, n, sigma, i as u64);
        let mod_cfg = QamModConfig {
            constellation: c,
            samples_per_symbol: 4,
            ..QamModConfig::default()
        };
        let make_tx =
            || QamModBlock::<VecStorage>::new(&mod_cfg, rate(40_000), &mut VecStorage).unwrap();
        let sent = run_block(&mut make_tx(), &bits, None);
        assert_eq!(sent.len(), 4 * bits.len() / mod_cfg.bits_per_symbol());
        check_block(
            &sent,
            &bits,
            SEED_CHUNKS,
            make_tx,
            &format!("qam_mod {c:?}"),
        );
        let cfg = QamDemodConfig {
            constellation: c,
            symbol_rate_hz: 10_000.0,
            equalizer_taps: taps,
            ..QamDemodConfig::default()
        };
        let make =
            || QamDemodBlock::<VecStorage>::new(&cfg, rate(40_000), &mut VecStorage).unwrap();
        let mut whole = make();
        let want = run_block(&mut whole, &input, None);
        assert!(
            want.len() > (n - 100) * cfg.bits_per_symbol(),
            "{c:?} taps {taps}"
        );
        assert!(whole.acquired_frequency().is_some(), "{c:?} taps {taps}");
        check_block(
            &want,
            &input,
            SEED_CHUNKS,
            make,
            &format!("qam_demod {c:?} taps {taps}"),
        );
    }
}
