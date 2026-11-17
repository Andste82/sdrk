//! The building blocks of M44.1 (tests 1 and 11): `symbol_map` (every point of every
//! constellation against the tables of `psk_mod` (the points at multiples of 22.5 degrees and
//! `PSK8_GRAY`) and of `Qam`, NaN and the sign rule, the differential phase and `pi4` over all
//! bit patterns, the invalid combinations, the timing, 20 cut plans of seed 840 and of seed
//! 851) and `iq_offset` (the delay by half a symbol of the imaginary and of the real part,
//! `reset`, the timing, the odd `samples_per_symbol`, 20 cut plans of seed 840 and of seed 851;
//! `samples_per_symbol` since M44.1.1 §3.3).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::trivially_copy_pass_by_ref
)]

mod common;

use common::{check_block, rate, run_block, same_bits};
use sdrk::{C32, Delay, Ratio, RatioKind};
use sdrk_blocks::{
    Block, Error, IqOffsetBlock, IqOffsetConfig, IqPart, PSK8_GRAY, SymbolMapBlock,
    SymbolMapConfig, SymbolMapConstellation, SymbolMapVariant,
};
use sdrk_core::kernels::constellation::Qam;
use sdrk_testkit::{Pcg32, VecStorage};

/// The seed of test 1 and of test 11.
const SEED: u64 = 840;
const SEED_CHUNKS: u64 = 851;

const ALL: [SymbolMapConstellation; 5] = [
    SymbolMapConstellation::Bpsk,
    SymbolMapConstellation::Qpsk,
    SymbolMapConstellation::Psk8,
    SymbolMapConstellation::Qam16,
    SymbolMapConstellation::Qam64,
];

fn map_block(cfg: &SymbolMapConfig) -> SymbolMapBlock<VecStorage> {
    SymbolMapBlock::new(cfg, rate(9600), &mut VecStorage).unwrap()
}

fn map_cfg(
    c: SymbolMapConstellation,
    differential: bool,
    variant: SymbolMapVariant,
) -> SymbolMapConfig {
    SymbolMapConfig {
        constellation: c,
        differential,
        variant,
    }
}

/// The soft bits of the symbol `s` of `bits` bits, the first the most significant: `+1` for a
/// 1, `-1` for a 0.
fn soft_of(s: usize, bits: usize) -> Vec<f32> {
    (0..bits)
        .map(|i| {
            if (s >> (bits - 1 - i)) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// The point at the phase `k pi / 8`, each part the `f32` of the exact value (the zeros of
/// the multiples of `pi / 2` exact).
fn point_at(k: u32) -> C32 {
    let x = f64::from(k % 16) * core::f64::consts::PI / 8.0;
    let part = |v: f64| if v.abs() < 1e-12 { 0.0 } else { v as f32 };
    C32::new(part(libm::cos(x)), part(libm::sin(x)))
}

/// The phase step (in units of `pi / 8`) of the symbol `s` of a PSK constellation: BPSK 1 -> 0,
/// 0 -> 8; QPSK Gray (`(2 b0 - 1, 2 b1 - 1) / sqrt(2)`); 8PSK `2 PSK8_GRAY[s] + 1`; pi/4-DQPSK
/// the steps of M40.1 §2 (`00` +pi/4, `01` +3 pi/4, `11` -3 pi/4, `10` -pi/4).
fn step_of(c: SymbolMapConstellation, pi4: bool, s: usize) -> u32 {
    match c {
        SymbolMapConstellation::Bpsk => {
            if s == 1 {
                0
            } else {
                8
            }
        }
        SymbolMapConstellation::Qpsk if pi4 => [2, 6, 14, 10][s],
        SymbolMapConstellation::Qpsk => [10, 6, 14, 2][s],
        _ => 2 * u32::from(PSK8_GRAY[s]) + 1,
    }
}

#[test]
fn symbol_map_gives_every_point_of_every_constellation() {
    let h = core::f32::consts::FRAC_1_SQRT_2;
    for c in ALL {
        let bits = c.bits_per_symbol();
        let n = 1usize << bits;
        let input: Vec<f32> = (0..n).flat_map(|s| soft_of(s, bits)).collect();
        let mut out = vec![C32::default(); n];
        let p = map_block(&map_cfg(c, false, SymbolMapVariant::Plain))
            .process(&input, &mut out)
            .unwrap();
        assert_eq!((p.consumed, p.produced), (n * bits, n), "{c:?}");
        for (s, y) in out.iter().enumerate() {
            let want = match c {
                SymbolMapConstellation::Qam16 => Qam::Qam16.point_table()[s],
                SymbolMapConstellation::Qam64 => Qam::Qam64.point_table()[s],
                _ => point_at(step_of(c, false, s)),
            };
            assert_eq!(y.re.to_bits(), want.re.to_bits(), "{c:?} symbol {s}");
            assert_eq!(y.im.to_bits(), want.im.to_bits(), "{c:?} symbol {s}");
        }
        // The plain points of M40 in their closed form.
        match c {
            SymbolMapConstellation::Bpsk => {
                assert_eq!(out, [C32::new(-1.0, 0.0), C32::new(1.0, 0.0)]);
            }
            SymbolMapConstellation::Qpsk => assert_eq!(
                out,
                [
                    C32::new(-h, -h),
                    C32::new(-h, h),
                    C32::new(h, -h),
                    C32::new(h, h)
                ]
            ),
            _ => {}
        }
    }
}

#[test]
fn symbol_map_follows_the_sign_rule() {
    // Above 0 is a 1; 0, -0, NaN and everything below 0 are a 0.
    let one = [1.0f32, f32::MIN_POSITIVE, 1e-30, f32::INFINITY, 0.5];
    let zero = [
        0.0f32,
        -0.0,
        f32::NAN,
        -f32::NAN,
        f32::NEG_INFINITY,
        -1e-30,
        -2.0,
    ];
    let mut b = map_block(&SymbolMapConfig::default());
    let mut out = [C32::default(); 1];
    for v in one {
        b.process(&[v], &mut out).unwrap();
        assert_eq!(out[0], C32::new(1.0, 0.0), "{v}");
    }
    for v in zero {
        b.process(&[v], &mut out).unwrap();
        assert_eq!(out[0], C32::new(-1.0, 0.0), "{v}");
    }
    // 16-QAM: the bits of the symbol 0b1010 written with extreme values.
    let mut q = map_block(&map_cfg(
        SymbolMapConstellation::Qam16,
        false,
        SymbolMapVariant::Plain,
    ));
    q.process(&[f32::INFINITY, f32::NAN, 1e-30, -0.0], &mut out)
        .unwrap();
    assert_eq!(out[0], Qam::Qam16.point_table()[0b1010]);
}

#[test]
fn symbol_map_differential_phase_over_all_bit_patterns() {
    for (c, pi4) in [
        (SymbolMapConstellation::Bpsk, false),
        (SymbolMapConstellation::Qpsk, false),
        (SymbolMapConstellation::Psk8, false),
        (SymbolMapConstellation::Qpsk, true),
    ] {
        let variant = if pi4 {
            SymbolMapVariant::Pi4
        } else {
            SymbolMapVariant::Plain
        };
        let bits = c.bits_per_symbol();
        let n = 1usize << bits;
        // Every pair of consecutive symbols, three times round.
        let symbols: Vec<usize> = (0..3 * n * n).map(|i| (i / n + i * 7) % n).collect();
        let input: Vec<f32> = symbols.iter().flat_map(|&s| soft_of(s, bits)).collect();
        let mut b = map_block(&map_cfg(c, true, variant));
        let got = run_block(&mut b, &input, None);
        let mut phase = 0u32;
        for (k, (&s, y)) in symbols.iter().zip(&got).enumerate() {
            phase = (phase + step_of(c, pi4, s)) % 16;
            assert_eq!(*y, point_at(phase), "{c:?} pi4 {pi4} symbol {k}");
        }
        // `reset` starts at phase 0 again.
        b.reset();
        let again = run_block(&mut b, &input, None);
        same_bits(&got, &again, "after reset");
    }
}

#[test]
fn symbol_map_rejects_the_invalid_combinations() {
    for c in [SymbolMapConstellation::Qam16, SymbolMapConstellation::Qam64] {
        let e = map_cfg(c, true, SymbolMapVariant::Plain)
            .validate()
            .unwrap_err();
        assert!(
            matches!(
                e,
                Error::InvalidParameter {
                    name: "differential",
                    ..
                }
            ),
            "{c:?}"
        );
    }
    for (c, differential) in [
        (SymbolMapConstellation::Qpsk, false),
        (SymbolMapConstellation::Bpsk, true),
        (SymbolMapConstellation::Psk8, true),
        (SymbolMapConstellation::Qam16, false),
    ] {
        let e = map_cfg(c, differential, SymbolMapVariant::Pi4)
            .validate()
            .unwrap_err();
        assert!(
            matches!(
                e,
                Error::InvalidParameter {
                    name: "variant",
                    ..
                }
            ),
            "{c:?} {differential}"
        );
        assert!(
            SymbolMapBlock::<VecStorage>::new(
                &map_cfg(c, differential, SymbolMapVariant::Pi4),
                rate(9600),
                &mut VecStorage
            )
            .is_err()
        );
    }
    for c in ALL {
        assert!(
            map_cfg(c, false, SymbolMapVariant::Plain)
                .validate()
                .is_ok()
        );
        assert!(
            map_cfg(c, c.is_psk(), SymbolMapVariant::Plain)
                .validate()
                .is_ok()
        );
    }
    assert!(
        map_cfg(SymbolMapConstellation::Qpsk, true, SymbolMapVariant::Pi4)
            .validate()
            .is_ok()
    );
}

#[test]
fn symbol_map_timing_and_whole_symbols() {
    for c in ALL {
        let bits = c.bits_per_symbol();
        let b = map_block(&map_cfg(c, false, SymbolMapVariant::Plain));
        let t = b.timing();
        assert_eq!(t.ratio, Ratio::new(1, bits as u64).unwrap(), "{c:?}");
        assert_eq!((t.kind, t.delay), (RatioKind::Constant, Delay::ZERO));
        assert_eq!(b.min_input(), bits);
        assert_eq!(
            SymbolMapBlock::<VecStorage>::rate_out(
                &map_cfg(c, false, SymbolMapVariant::Plain),
                rate(9600)
            )
            .unwrap(),
            sdrk::Rate::new(9600, bits as u64).unwrap()
        );
    }
    // 8PSK: 7 bits are two symbols, the seventh waits; one output sample takes one symbol.
    let mut b = map_block(&map_cfg(
        SymbolMapConstellation::Psk8,
        false,
        SymbolMapVariant::Plain,
    ));
    let mut out = [C32::default(); 4];
    let p = b.process(&[1.0; 7], &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (6, 2));
    let p = b.process(&[1.0; 7], &mut out[..1]).unwrap();
    assert_eq!((p.consumed, p.produced), (3, 1));
    let p = b.process(&[1.0; 7], &mut []).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
}

/// Random soft bits of `seed` and `stream` with the extreme values of the sign rule mixed in.
fn soft_bits(n: usize, seed: u64, stream: u64) -> Vec<f32> {
    let mut rng = Pcg32::new(seed, stream);
    (0..n)
        .map(|i| match (rng.next_u32() % 2, i % 41) {
            (_, 40) => f32::NAN,
            (_, 39) => -0.0,
            (1, _) => 1.0,
            _ => -1.0,
        })
        .collect()
}

#[test]
fn symbol_map_is_chunk_invariant() {
    let mut stream = 0;
    for c in ALL {
        for (differential, variant) in [
            (false, SymbolMapVariant::Plain),
            (true, SymbolMapVariant::Plain),
            (true, SymbolMapVariant::Pi4),
        ] {
            let cfg = map_cfg(c, differential, variant);
            if cfg.validate().is_err() {
                continue;
            }
            stream += 1;
            let input = soft_bits(6000, SEED, stream);
            let want = run_block(&mut map_block(&cfg), &input, None);
            assert_eq!(want.len(), 6000 / c.bits_per_symbol());
            for seed in [SEED, SEED_CHUNKS] {
                check_block(&want, &input, seed, || map_block(&cfg), &format!("{cfg:?}"));
            }
        }
    }
}

fn offset_block(sps: usize, part: IqPart) -> IqOffsetBlock<VecStorage> {
    IqOffsetBlock::new(
        &IqOffsetConfig {
            samples_per_symbol: sps,
            part,
        },
        rate(9600),
        &mut VecStorage,
    )
    .unwrap()
}

/// A random complex signal of `seed` and `stream`.
fn complex(n: usize, seed: u64, stream: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(seed, stream);
    let mut u = || (f64::from(rng.next_u32()) / 2f64.powi(31) - 1.0) as f32;
    (0..n).map(|_| C32::new(u(), u())).collect()
}

#[test]
fn iq_offset_delays_one_part() {
    for (stream, sps) in [2usize, 4, 8, 10, 64, 1024].into_iter().enumerate() {
        let samples = sps / 2;
        let x = complex(3000, SEED, 100 + stream as u64);
        for part in [IqPart::Imag, IqPart::Real] {
            let mut b = offset_block(sps, part);
            let y = run_block(&mut b, &x, None);
            assert_eq!(y.len(), x.len());
            for (n, (a, o)) in x.iter().zip(&y).enumerate() {
                let late = if n >= samples {
                    x[n - samples]
                } else {
                    C32::default()
                };
                let want = match part {
                    IqPart::Imag => C32::new(a.re, late.im),
                    _ => C32::new(late.re, a.im),
                };
                assert_eq!(o.re.to_bits(), want.re.to_bits(), "{samples} {part:?} {n}");
                assert_eq!(o.im.to_bits(), want.im.to_bits(), "{samples} {part:?} {n}");
            }
            // `reset`: zeros in the ring and the position at its start again.
            b.reset();
            same_bits(&y, &run_block(&mut b, &x, None), "after reset");
            // In place, in pieces of 7 samples, the same.
            b.reset();
            let mut z = x.clone();
            for piece in z.chunks_mut(7) {
                b.process_in_place(piece);
            }
            same_bits(&y, &z, "in place");
            assert_eq!(
                b.config(),
                IqOffsetConfig {
                    samples_per_symbol: sps,
                    part
                }
            );
            let t = b.timing();
            assert_eq!(
                (t.ratio, t.kind, t.delay),
                (Ratio::ONE, RatioKind::Constant, Delay::ZERO)
            );
            assert_eq!(b.min_input(), 1);
        }
    }
    // A call takes min(input, output) samples.
    let mut b = offset_block(6, IqPart::Imag);
    let p = b
        .process(&complex(10, SEED, 1), &mut [C32::default(); 4])
        .unwrap();
    assert_eq!((p.consumed, p.produced), (4, 4));
    for sps in [0, 1026] {
        assert!(matches!(
            IqOffsetConfig {
                samples_per_symbol: sps,
                part: IqPart::Imag
            }
            .validate(),
            Err(Error::OutOfRange {
                name: "samples_per_symbol",
                ..
            })
        ));
    }
    // The delay is half a symbol: an odd `samples_per_symbol` is refused, by the configuration
    // and by `set`.
    for sps in [3usize, 7, 1023] {
        let mut cfg = IqOffsetConfig {
            samples_per_symbol: sps,
            part: IqPart::Imag,
        };
        assert!(matches!(
            cfg.validate(),
            Err(Error::InvalidParameter {
                name: "samples_per_symbol",
                ..
            })
        ));
        assert!(IqOffsetBlock::new(&cfg, rate(9600), &mut VecStorage).is_err());
        cfg.samples_per_symbol = 8;
        assert!(matches!(
            cfg.set(
                "samples_per_symbol",
                sdrk_blocks::ParamValue::I64(sps as i64)
            ),
            Err(Error::InvalidParameter {
                name: "samples_per_symbol",
                ..
            })
        ));
        assert_eq!(cfg.samples_per_symbol, 8);
    }
}

#[test]
fn iq_offset_is_chunk_invariant() {
    for (stream, sps) in [2usize, 6, 8, 510, 1024].into_iter().enumerate() {
        let x = complex(5000, SEED, 200 + stream as u64);
        for part in [IqPart::Imag, IqPart::Real] {
            let want = run_block(&mut offset_block(sps, part), &x, None);
            for seed in [SEED, SEED_CHUNKS] {
                check_block(
                    &want,
                    &x,
                    seed,
                    || offset_block(sps, part),
                    &format!("iq_offset {sps} {part:?}"),
                );
            }
        }
    }
}
