//! The blocks of M40 (test 6 and the block rules): `CostasBlock` (the kernel `costas` through
//! the facade), `PskModBlock` (the Gray points, the differential phase, the pulse through
//! `fir_interpolate`) and `PskDemodBlock` (the composition `fir` -> `symbol_sync` -> `costas`
//! or the differential product -> soft bits), each equal to its kernels whole and in 20 cut
//! plans of seed 559; the timing, rates and parameter checks; the extremes (NaN, infinity,
//! silence, full scale); the loop `psk_mod` -> `psk_demod` without noise is error-free for
//! BPSK and QPSK, coherent and differential; `frame_decode` with `polarity` both.
#![allow(
    clippy::approx_constant,
    clippy::assert_is_empty,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

mod common;

use common::{check_block, rate, run_block, same_bits};
use sdrk::{C32, Delay, Ratio, RatioKind, Sdrk};
use sdrk_blocks::design::{rrc, rrc_len};
use sdrk_blocks::{
    Block, CostasBlock, CostasConfig, Error, FrameDecodeBlock, FrameDecodeConfig, FramePolarity,
    Packet, PacketFlags, ParamValue, PskDemodBlock, PskDemodConfig, PskModBlock, PskModConfig,
};
use sdrk_core::kernels::costas::{Costas, CostasState};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::interp::{FirInterpState, FirInterpolate};
use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncState, Ted};
use sdrk_device::Rate;
use sdrk_testkit::{Pcg32, VecStorage};

/// Seed of the chunk-invariance tests of M40 (test 6) and of the random signals of this file.
const SEED: u64 = 559;

const H: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// Random soft bits `+-1` (stream `stream` of the seed).
fn soft_bits(n: usize, stream: u64) -> Vec<f32> {
    let mut rng = Pcg32::new(SEED, stream);
    (0..n)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

/// The point at `k pi / 4`.
fn point(k: u8) -> C32 {
    [
        C32::new(1.0, 0.0),
        C32::new(H, H),
        C32::new(0.0, 1.0),
        C32::new(-H, H),
        C32::new(-1.0, 0.0),
        C32::new(-H, -H),
        C32::new(0.0, -1.0),
        C32::new(H, -H),
    ][usize::from(k & 7)]
}

/// The symbols of §3, written out independently: Gray mapping, the differential phase in
/// units of 45 degrees.
fn symbols(bits: &[f32], order: u32, differential: bool) -> Vec<C32> {
    let bps = if order == 4 { 2 } else { 1 };
    let mut phase = 0u8;
    bits.chunks_exact(bps)
        .map(|b| {
            let k = if bps == 1 {
                if b[0] > 0.0 { 0 } else { 4 }
            } else {
                match (b[0] > 0.0, b[1] > 0.0) {
                    (true, true) => 1,
                    (false, true) => 3,
                    (false, false) => 5,
                    (true, false) => 7,
                }
            };
            if differential {
                phase = (phase + k) & 7;
                point(phase)
            } else {
                point(k)
            }
        })
        .collect()
}

fn rrc_taps(rolloff: f64, sps: usize, span: usize, gain: f64) -> Vec<f32> {
    let mut h = vec![0.0f64; rrc_len(sps, span).unwrap()];
    rrc(rolloff, sps, span, &mut h).unwrap();
    h.iter().map(|v| (v * gain) as f32).collect()
}

/// `psk_mod` composed from the kernels in one call.
fn mod_kernels(cfg: &PskModConfig, bits: &[f32]) -> Vec<C32> {
    let sps = cfg.samples_per_symbol;
    let taps = rrc_taps(cfg.rolloff, sps, cfg.span, (sps as f64).sqrt());
    let n = taps.len();
    let mut st =
        FirInterpState::<C32, _, _>::new(taps, vec![C32::default(); n.div_ceil(sps)], sps).unwrap();
    let syms = symbols(bits, cfg.order, cfg.differential);
    let mut out = vec![C32::default(); syms.len() * sps];
    let p =
        <Sdrk as FirInterpolate<C32>>::fir_interpolate(&Sdrk, &mut st, &syms, &mut out).unwrap();
    assert_eq!(p.consumed, syms.len());
    out
}

/// `psk_demod` composed from the kernels in one call each.
fn demod_kernels(cfg: &PskDemodConfig, fs: Rate, input: &[C32]) -> Vec<f32> {
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
    let mut y = vec![C32::default(); syms.len()];
    if cfg.differential {
        let mut prev = C32::new(1.0, 0.0);
        for (z, &s) in y.iter_mut().zip(&syms) {
            *z = C32::new(
                s.re * prev.re + s.im * prev.im,
                s.im * prev.re - s.re * prev.im,
            );
            prev = s;
        }
    } else {
        let mut st = CostasState::new(cfg.order, cfg.carrier_bw, 0.7071, cfg.max_offset).unwrap();
        <Sdrk as Costas>::costas(&Sdrk, &mut st, &syms, &mut y).unwrap();
    }
    let mut bits = Vec::new();
    for s in y {
        bits.push(s.re);
        if cfg.order == 4 {
            bits.push(s.im);
        }
    }
    bits
}

/// A noisy, turned and frequency-shifted QPSK or BPSK signal at `sps` samples per symbol from
/// `psk_mod` (the input of the demodulator tests).
fn received(order: u32, differential: bool, sps: usize, n_sym: usize, stream: u64) -> Vec<C32> {
    let cfg = PskModConfig {
        order,
        samples_per_symbol: sps,
        differential,
        ..PskModConfig::default()
    };
    let bps = cfg.bits_per_symbol();
    let tx = mod_kernels(&cfg, &soft_bits(n_sym * bps, stream));
    let mut rng = Pcg32::new(SEED, 100 + stream);
    tx.iter()
        .enumerate()
        .map(|(i, x)| {
            let ph = 0.6 + 2.0 * std::f64::consts::PI * 0.001 * i as f64 / sps as f64;
            let (s, c) = ph.sin_cos();
            let (re, im) = (f64::from(x.re), f64::from(x.im));
            C32::new(
                (re * c - im * s + 0.05 * rng.normal()) as f32,
                (re * s + im * c + 0.05 * rng.normal()) as f32,
            )
        })
        .collect()
}

// ---- costas ------------------------------------------------------------------------------

#[test]
fn costas_block_equals_the_kernel_whole_and_in_cut_plans() {
    for order in [2u32, 4] {
        let cfg = CostasConfig {
            order,
            loop_bw: 0.03,
            ..CostasConfig::default()
        };
        // one sample per symbol: the symbols of `psk_mod`, turned and noisy
        let bits = soft_bits(6000, u64::from(order));
        let mut rng = Pcg32::new(SEED, 10 + u64::from(order));
        let input: Vec<C32> = symbols(&bits, order, false)
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let (sn, cs) = (1.0 + 0.004 * k as f32).sin_cos();
                C32::new(
                    s.re * cs - s.im * sn + 0.1 * rng.normal() as f32,
                    s.re * sn + s.im * cs + 0.1 * rng.normal() as f32,
                )
            })
            .collect();
        let mut st = CostasState::new(order, 0.03, 0.7071, 0.05).unwrap();
        let mut want = vec![C32::default(); input.len()];
        <Sdrk as Costas>::costas(&Sdrk, &mut st, &input, &mut want).unwrap();
        check_block(
            &want,
            &input,
            SEED,
            || CostasBlock::<VecStorage>::new(&cfg, rate(1000), &mut VecStorage).unwrap(),
            &format!("costas order {order}"),
        );
        let mut block = CostasBlock::<VecStorage>::new(&cfg, rate(1000), &mut VecStorage).unwrap();
        let got = run_block(&mut block, &input, None);
        assert_eq!(got.len(), input.len());
        assert_eq!(block.state(), &st);
        assert!(block.locked(), "order {order}");
        assert!(
            (block.frequency() - 0.004 / (2.0 * std::f64::consts::PI)).abs() < 2e-4,
            "order {order}: {}",
            block.frequency()
        );
        let t = block.timing();
        assert_eq!(
            (t.ratio, t.kind, t.delay),
            (Ratio::ONE, RatioKind::Constant, Delay::ZERO)
        );
        assert_eq!(block.min_input(), 1);
        block.reset();
        assert_eq!(
            block.state(),
            &CostasState::new(order, 0.03, 0.7071, 0.05).unwrap()
        );
        // a short output takes as many symbols as fit
        let p = block
            .process(&input[..10], &mut [C32::default(); 4])
            .unwrap();
        assert_eq!((p.consumed, p.produced), (4, 4));
    }
}

#[test]
fn costas_block_parameters() {
    let ok = CostasConfig::default();
    assert!(ok.validate().is_ok());
    let bad = [
        CostasConfig { order: 3, ..ok },
        CostasConfig { order: 8, ..ok },
        CostasConfig {
            loop_bw: 0.00009,
            ..ok
        },
        CostasConfig {
            loop_bw: 0.21,
            ..ok
        },
        CostasConfig {
            damping: 0.05,
            ..ok
        },
        CostasConfig {
            max_offset: -0.01,
            ..ok
        },
        CostasConfig {
            max_offset: 0.26,
            ..ok
        },
        CostasConfig {
            loop_bw: f64::NAN,
            ..ok
        },
    ];
    for cfg in bad {
        assert!(
            CostasBlock::<VecStorage>::new(&cfg, rate(1000), &mut VecStorage).is_err(),
            "{cfg:?}"
        );
    }
    let mut b = CostasBlock::<VecStorage>::new(&ok, rate(1000), &mut VecStorage).unwrap();
    assert_eq!(
        b.set_param("loop_bw", ParamValue::F64(0.01)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(b.param("max_offset"), Ok(ParamValue::F64(0.05)));
}

// ---- psk_mod -----------------------------------------------------------------------------

#[test]
fn psk_mod_maps_gray_symbols_and_the_differential_phase() {
    // BPSK and QPSK, coherent and differential, at 4 samples per symbol and the default span:
    // the block equals the kernel chain (`mod_kernels`), whole and in cut plans; the symbol
    // mapping itself is checked against the explicit tables below.
    for (order, differential) in [(2, false), (2, true), (4, false), (4, true)] {
        let cfg = PskModConfig {
            order,
            samples_per_symbol: 4,
            differential,
            ..PskModConfig::default()
        };
        let bits = soft_bits(4000, 20 + u64::from(order) + u64::from(differential));
        let want = mod_kernels(&cfg, &bits);
        check_block(
            &want,
            &bits,
            SEED,
            || PskModBlock::<VecStorage>::new(&cfg, rate(9600), &mut VecStorage).unwrap(),
            &format!("psk_mod order {order} differential {differential}"),
        );
    }
    // The explicit table: BPSK 1 -> +1, 0 -> -1; QPSK (b0, b1) -> ((2 b0 - 1), (2 b1 - 1)) / sqrt 2.
    assert_eq!(
        symbols(&[1.0, -1.0, f32::NAN, 0.0], 2, false),
        [
            C32::new(1.0, 0.0),
            C32::new(-1.0, 0.0),
            C32::new(-1.0, 0.0),
            C32::new(-1.0, 0.0)
        ]
    );
    assert_eq!(
        symbols(&[1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0], 4, false),
        [
            C32::new(H, H),
            C32::new(-H, H),
            C32::new(-H, -H),
            C32::new(H, -H)
        ]
    );
    // Differential BPSK: a 1 keeps the phase, a 0 flips it.
    assert_eq!(
        symbols(&[1.0, -1.0, -1.0, 1.0], 2, true),
        [
            C32::new(1.0, 0.0),
            C32::new(-1.0, 0.0),
            C32::new(1.0, 0.0),
            C32::new(1.0, 0.0)
        ]
    );
    // Differential QPSK: (1, 1) adds 45 degrees each time; eight of them close the circle.
    let q = symbols(&[1.0; 16], 4, true);
    assert_eq!(q[1], C32::new(0.0, 1.0));
    assert_eq!(q[7], C32::new(1.0, 0.0));
}

#[test]
fn psk_mod_symbol_centres_and_timing() {
    // An isolated symbol: the output is the pulse times the symbol, its peak sqrt(sps) h[c].
    let cfg = PskModConfig {
        order: 4,
        samples_per_symbol: 8,
        span: 10,
        ..PskModConfig::default()
    };
    let r = rate(50_000);
    let mut block = PskModBlock::<VecStorage>::new(&cfg, r, &mut VecStorage).unwrap();
    assert_eq!(block.num_taps(), 81);
    let t = block.timing();
    assert_eq!(t.ratio, Ratio::new(4, 1).unwrap());
    assert_eq!(t.kind, RatioKind::Constant);
    assert_eq!(t.delay, Delay::new(20, 2).unwrap()); // span / 2 symbols of 2 bits
    assert_eq!(block.min_input(), 2);
    assert_eq!(
        PskModBlock::<VecStorage>::rate_out(&cfg, r).unwrap(),
        Rate::hz(200_000).unwrap()
    );
    let bits = [1.0f32, 1.0];
    let mut out = vec![C32::default(); 8];
    let p = block.process(&bits, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (2, 8));
    let taps = rrc_taps(0.35, 8, 10, 8f64.sqrt());
    for (k, o) in out.iter().enumerate() {
        assert_eq!(*o, C32::new(H * taps[k], H * taps[k]), "sample {k}");
    }
    // Whole symbols only: an odd bit, or less output room than a symbol, stays unconsumed.
    let p = block.process(&[1.0], &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    let p = block.process(&[1.0, 1.0], &mut out[..7]).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    // reset: the history and the differential phase start again
    block.reset();
    let mut again = vec![C32::default(); 8];
    block.process(&bits, &mut again).unwrap();
    assert_eq!(again, out);
    // parameters
    for bad in [
        PskModConfig {
            samples_per_symbol: 1,
            ..cfg
        },
        PskModConfig {
            samples_per_symbol: 65,
            ..cfg
        },
        PskModConfig { span: 3, ..cfg },
        PskModConfig { span: 33, ..cfg },
        PskModConfig {
            rolloff: 0.04,
            ..cfg
        },
        PskModConfig {
            rolloff: 1.01,
            ..cfg
        },
        PskModConfig { order: 8, ..cfg },
    ] {
        assert!(
            PskModBlock::<VecStorage>::new(&bad, r, &mut VecStorage).is_err(),
            "{bad:?}"
        );
    }
    assert_eq!(
        block.set_param("span", ParamValue::I64(12)),
        Err(Error::ReadOnlyParameter)
    );
}

// ---- psk_demod ---------------------------------------------------------------------------

#[test]
fn psk_demod_equals_the_kernels_whole_and_in_cut_plans() {
    for (i, (order, differential, sps)) in [
        (2u32, false, 4usize),
        (4, false, 8),
        (2, true, 2),
        (4, true, 5),
    ]
    .into_iter()
    .enumerate()
    {
        let fs = rate(9600 * sps as u64);
        let cfg = PskDemodConfig {
            order,
            symbol_rate_hz: 9600.0,
            differential,
            ..PskDemodConfig::default()
        };
        let input = received(order, differential, sps, 3000, 40 + i as u64);
        let want = demod_kernels(&cfg, fs, &input);
        assert!(want.len() > 2900 * cfg.bits_per_symbol(), "case {i}");
        check_block(
            &want,
            &input,
            SEED,
            || PskDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap(),
            &format!("psk_demod case {i}"),
        );
    }
}

#[test]
fn psk_demod_keeps_soft_bits_that_do_not_fit() {
    let fs = rate(38_400);
    let cfg = PskDemodConfig {
        order: 4,
        symbol_rate_hz: 9600.0,
        ..PskDemodConfig::default()
    };
    let input = received(4, false, 4, 2000, 60);
    let want = demod_kernels(&cfg, fs, &input);
    let mut block = PskDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
    for room_len in [1usize, 3, 5] {
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
        assert!(rest.is_empty());
        same_bits(&want, &got, &format!("room {room_len}"));
    }
    // no output room: nothing is consumed
    block.reset();
    let p = block.process(&input, &mut []).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
}

#[test]
fn psk_demod_timing_rates_and_parameters() {
    let cfg = PskDemodConfig {
        order: 4,
        symbol_rate_hz: 25_000.0,
        ..PskDemodConfig::default()
    };
    let fs = rate(200_000);
    let block = PskDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
    assert_eq!(block.samples_per_symbol(), 8);
    assert_eq!(block.num_taps(), 81);
    let t = block.timing();
    assert_eq!(t.ratio, Ratio::new(1, 4).unwrap());
    assert_eq!(t.kind, RatioKind::Variable);
    assert_eq!(t.delay, Delay::new(82, 2).unwrap()); // 40 of the filter plus 1
    assert_eq!(
        block.min_input(),
        (8.0f64 * 1.005 * 1.25).ceil() as usize + 3
    );
    assert_eq!(
        PskDemodBlock::<VecStorage>::rate_out(&cfg, fs).unwrap(),
        Rate::hz(50_000).unwrap()
    );
    assert!(!block.locked());
    // within 1 % of an integer: 8.07 samples per symbol is accepted (the filter has 8, the
    // timing loop the exact value)
    let near = PskDemodConfig {
        symbol_rate_hz: 200_000.0 / 8.07,
        ..cfg
    };
    let b = PskDemodBlock::<VecStorage>::new(&near, fs, &mut VecStorage).unwrap();
    assert_eq!(b.samples_per_symbol(), 8);
    assert!((b.sps_estimate() - 8.07).abs() < 1e-6);
    // 2 and 2.01 samples per symbol are the lowest rates accepted (D625)
    for sps in [2.0, 2.01] {
        let low = PskDemodConfig {
            symbol_rate_hz: 200_000.0 / sps,
            ..cfg
        };
        assert_eq!(low.samples_per_symbol(fs).unwrap(), 2, "{sps}");
    }
    for sym in [
        200_000.0 / 8.1,
        200_000.0 / 7.5,
        200_000.0 / 1.5,
        // within 1 % of 2 but below 2 samples per symbol: the timing loop needs at least 2,
        // reported by the same rate error (D625)
        200_000.0 / 1.99,
        200_000.0 / 1.981,
        150_000.0,
        1.0,
    ] {
        let c = PskDemodConfig {
            symbol_rate_hz: sym,
            ..cfg
        };
        match PskDemodBlock::<VecStorage>::new(&c, fs, &mut VecStorage) {
            Err(Error::InvalidParameter { name, reason }) => {
                assert_eq!(name, "symbol_rate_hz");
                assert!(reason.contains("resample"), "{reason}");
            }
            other => panic!("{sym}: {other:?}"),
        }
    }
    for bad in [
        PskDemodConfig { order: 3, ..cfg },
        PskDemodConfig { span: 33, ..cfg },
        PskDemodConfig {
            rolloff: 0.01,
            ..cfg
        },
        PskDemodConfig {
            sync_bw: 0.2,
            ..cfg
        },
        PskDemodConfig {
            carrier_bw: 0.3,
            ..cfg
        },
        PskDemodConfig {
            max_offset: 0.3,
            ..cfg
        },
    ] {
        assert!(
            PskDemodBlock::<VecStorage>::new(&bad, fs, &mut VecStorage).is_err(),
            "{bad:?}"
        );
    }
    let bpsk = PskDemodConfig { order: 2, ..cfg };
    assert_eq!(
        PskDemodBlock::<VecStorage>::new(&bpsk, fs, &mut VecStorage)
            .unwrap()
            .timing()
            .ratio,
        Ratio::new(1, 8).unwrap()
    );
}

// ---- the loop ----------------------------------------------------------------------------

/// The lag (0 to 64 bits) at which the demodulated hard bits equal the sent bits from bit 400
/// on, and the number of errors there.
fn best_lag(sent: &[f32], soft: &[f32]) -> (usize, usize) {
    (0..64)
        .map(|lag| {
            let errors = soft
                .iter()
                .skip(400 + lag)
                .zip(sent.iter().skip(400))
                .filter(|(s, b)| (**s > 0.0) != (**b > 0.0))
                .count();
            (lag, errors)
        })
        .min_by_key(|&(_, e)| e)
        .unwrap()
}

#[test]
fn psk_mod_to_psk_demod_without_noise_is_error_free() {
    for (order, differential) in [(2u32, false), (2, true), (4, false), (4, true)] {
        let sps = 4;
        let m = PskModConfig {
            order,
            samples_per_symbol: sps,
            differential,
            ..PskModConfig::default()
        };
        let d = PskDemodConfig {
            order,
            symbol_rate_hz: 1000.0,
            differential,
            ..PskDemodConfig::default()
        };
        let bits = soft_bits(6000, 70 + u64::from(order) + u64::from(differential));
        let bit_rate = rate(1000 * m.bits_per_symbol() as u64);
        let mut tx = PskModBlock::<VecStorage>::new(&m, bit_rate, &mut VecStorage).unwrap();
        let mut rx = PskDemodBlock::<VecStorage>::new(&d, rate(4000), &mut VecStorage).unwrap();
        let sig = run_block(&mut tx, &bits, None);
        assert_eq!(sig.len(), bits.len() / m.bits_per_symbol() * sps);
        let soft = run_block(&mut rx, &sig, None);
        let (lag, errors) = best_lag(&bits, &soft);
        assert_eq!(
            errors, 0,
            "order {order} differential {differential}: lag {lag}"
        );
        // the symbol centres have amplitude 1 (coherent; differential |y|^2 = 1) up to the
        // intersymbol interference of the truncated pulse and the timing jitter
        let tail = &soft[soft.len() - 200..];
        let want = if order == 4 { H } else { 1.0 };
        for s in tail {
            assert!(
                (s.abs() - want).abs() < 0.1 * want,
                "{order} {differential}: {s}"
            );
        }
        if !differential {
            assert!(rx.locked());
        }
    }
}

#[test]
fn extremes_nan_infinity_silence_and_full_scale() {
    // psk_mod: NaN and infinite soft bits are bits (NaN and -inf a 0, +inf a 1); the output is
    // finite.
    let cfg = PskModConfig {
        order: 4,
        samples_per_symbol: 4,
        ..PskModConfig::default()
    };
    let mut m = PskModBlock::<VecStorage>::new(&cfg, rate(2000), &mut VecStorage).unwrap();
    let odd = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::MAX,
    ];
    let out = run_block(&mut m, &odd, None);
    assert_eq!(out.len(), 12);
    assert!(out.iter().all(|c| c.re.is_finite() && c.im.is_finite()));
    let mut m2 = PskModBlock::<VecStorage>::new(&cfg, rate(2000), &mut VecStorage).unwrap();
    let same = run_block(&mut m2, &[-1.0, 1.0, -1.0, -1.0, -1.0, 1.0], None);
    assert_eq!(out, same);

    // psk_demod: silence gives zero soft bits; a burst of NaN and infinity in the middle of a
    // clean signal poisons only its neighbourhood, and the bits after it are right again.
    for (order, differential) in [(2u32, false), (4, false), (4, true)] {
        let d = PskDemodConfig {
            order,
            symbol_rate_hz: 1000.0,
            differential,
            ..PskDemodConfig::default()
        };
        let mut rx = PskDemodBlock::<VecStorage>::new(&d, rate(4000), &mut VecStorage).unwrap();
        let silence = run_block(&mut rx, &vec![C32::default(); 4000], None);
        assert!(silence.len() > 990 * d.bits_per_symbol());
        assert!(silence.iter().all(|s| *s == 0.0), "{order} {differential}");
        assert!(!rx.locked());
        let m = PskModConfig {
            order,
            samples_per_symbol: 4,
            differential,
            ..PskModConfig::default()
        };
        let bits = soft_bits(8000, 80 + u64::from(order));
        let mut tx = PskModBlock::<VecStorage>::new(&m, rate(1000), &mut VecStorage).unwrap();
        let mut sig = run_block(&mut tx, &bits, None);
        let mid = sig.len() / 2;
        for (k, x) in sig[mid..mid + 40].iter_mut().enumerate() {
            *x = match k % 4 {
                0 => C32::new(f32::NAN, 0.0),
                1 => C32::new(f32::INFINITY, -f32::INFINITY),
                2 => C32::new(f32::MAX, f32::MAX),
                _ => C32::new(0.0, f32::NAN),
            };
        }
        // full scale (amplitude 1 at the symbol centres) is the normal level of the loop
        rx.reset();
        let soft = run_block(&mut rx, &sig, None);
        assert!(soft.len() > (bits.len() - 64) * 9 / 10);
        // bits well after the burst are right again (coherent loops keep their phase: the
        // detector and lock rules skip non-finite symbols)
        let after = mid / 4 * m.bits_per_symbol() + 400;
        let (lag, _) = best_lag(&bits[..after / 2], &soft[..after / 2]);
        let errors = soft
            .iter()
            .skip(after + lag)
            .zip(bits.iter().skip(after))
            .filter(|(s, b)| (**s > 0.0) != (**b > 0.0))
            .count();
        assert_eq!(errors, 0, "order {order} differential {differential}");
        assert!(soft[after + lag..].iter().all(|s| s.is_finite()));
    }
}

// ---- frame_decode polarity ---------------------------------------------------------------

#[test]
fn frame_decode_with_both_polarities_reads_inverted_frames() {
    let sent = Packet::new(0, PacketFlags::NONE, b"polarity").unwrap();
    let mut bits = vec![0u8; sdrk_proto::frame::frame_bits(8, 4)];
    sdrk_proto::frame::encode(&sent, 4, &mut bits).unwrap();
    let normal: Vec<f32> = bits
        .iter()
        .map(|&b| if b == 1 { 1.0 } else { -1.0 })
        .collect();
    let inverted: Vec<f32> = normal.iter().map(|x| -x).collect();
    let decode = |polarity, soft: &[f32]| {
        let cfg = FrameDecodeConfig {
            polarity,
            ..FrameDecodeConfig::default()
        };
        let mut dec =
            FrameDecodeBlock::<VecStorage>::new(&cfg, rate(9600), &mut VecStorage).unwrap();
        let mut out = vec![Packet::default(); 4];
        let p = dec.process(soft, &mut out).unwrap();
        out.truncate(p.produced);
        out
    };
    // normal: only the sent polarity
    assert_eq!(decode(FramePolarity::Normal, &normal).len(), 1);
    assert!(decode(FramePolarity::Normal, &inverted).is_empty());
    // both: either polarity, same packet and index
    for soft in [&normal, &inverted] {
        let got = decode(FramePolarity::Both, soft);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].data(), b"polarity");
        assert_eq!(got[0].flags, PacketFlags::CRC_OK);
        assert_eq!(got[0].index, 32);
    }
    // a normal frame after an inverted one: the polarity is found per frame
    let mut two = inverted.clone();
    two.extend_from_slice(&normal);
    let got = decode(FramePolarity::Both, &two);
    assert_eq!(got.len(), 2);
    assert_eq!(got[1].index, (normal.len() + 32) as u64);
    assert!(got.iter().all(|p| p.data() == b"polarity"));
}
