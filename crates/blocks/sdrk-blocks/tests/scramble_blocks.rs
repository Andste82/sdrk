//! The block of M49: `ScrambleBlock` (`C32`, `CQ15`) of the kind `scramble` (M49 §6, test 6):
//! equality with the kernel `scramble` (through `Sdrk`) on the PRBS indices built bit by bit
//! with `Prbs::next_bit` (MSB first), on whole buffers and in 20 chunk plans (seed 696), for
//! every key and every `bits`; the round trip scrambler then descrambler (exact at `bits` 1 and
//! 2, within the bounds of §3 at `bits` 3 to 8); `reset`, the call limit `SCRAMBLE_CHUNK`, the
//! description rules of M09, validation at the parameter, exact storage (none), timing, size,
//! `Send` and `Sync`. (The word-parallel PRBS step against `next_bit` over the full period is
//! `sdrk-proto/tests/prbs_m49.rs`.)
//!
//! M49.1 test 4 (seed 870): the word-parallel index production of the block equals the
//! per-bit specification (`next_bit`, MSB first) over the full period of every polynomial at
//! every `bits` 1 to 8, on the whole buffer and in 20 chunk plans; the indices are read back
//! from the block output for the probe input `1 + 0j`, whose output is the phasor of the index.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod common;

use common::{check_exact_storage, rate, run_block, twenty_plans};
use sdrk::{C32, CQ15, Delay, Q15, Ratio, RatioKind, Sdrk};
use sdrk_blocks::{
    ArenaStorage, Block, Error, Mutability, ParamKind, ParamValue, PortType, SCRAMBLE_CHUNK,
    ScrambleBlock, ScrambleConfig, Unit, all_descriptions,
};
use sdrk_core::kernels::scramble::Scramble;
use sdrk_proto::prbs::{Poly, Prbs};
use sdrk_testkit::{Pcg32, VecStorage};

/// Seed of the signals and of the chunk plans (M49 range 690 to 699; test 6 takes 696, 697).
const SEED: u64 = 696;
/// Seed of the M49.1 index production test (M49.1 range 870 to 879; test 4 takes 870).
const SEED_M49_1: u64 = 870;
/// Samples per test (more than one period of `prbs9` at `bits` 8, several calls of 256).
const N: usize = 3000;

/// Every key polynomial.
const KEYS: [Poly; 6] = [
    Poly::Prbs5,
    Poly::Prbs7,
    Poly::Prbs9,
    Poly::Prbs10,
    Poly::Prbs11,
    Poly::Prbs15,
];

fn cfg(bits: u32, inverse: bool, key: Poly, seed: u32) -> ScrambleConfig {
    ScrambleConfig {
        bits,
        inverse,
        key,
        seed,
    }
}

/// The phase indices of `n` samples: `bits` bits of `Prbs::next_bit` per sample, the first one
/// as the most significant bit (the specification of §6).
fn indices(key: Poly, seed: u32, bits: u32, n: usize) -> Vec<u8> {
    let mut p = Prbs::new(key, seed).unwrap();
    (0..n)
        .map(|_| (0..bits).fold(0u8, |w, _| (w << 1) | p.next_bit()))
        .collect()
}

/// The kernel output for the configuration, through the facade.
fn kernel<T>(c: &ScrambleConfig, input: &[T]) -> Vec<T>
where
    Sdrk: Scramble<T>,
    T: sdrk_core::ComplexSample + Copy + Default,
{
    let phase = indices(c.key, c.seed, c.bits, input.len());
    let mut out = vec![T::default(); input.len()];
    Sdrk.scramble(input, &phase, c.bits, c.inverse, &mut out)
        .unwrap();
    out
}

/// The raw bytes of a sample buffer (bit comparison, NaN payloads and `±0` included).
fn bytes<T: bytemuck::Pod>(v: &[T]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

/// A `C32` signal in the unit disk with special values: `±0`, and (with
/// `specials`) `(1, −1)`, NaN, `±∞` and subnormals.
fn c32_signal(n: usize, seed: u64, specials: bool) -> Vec<C32> {
    let mut rng = Pcg32::new(seed, 3);
    (0..n)
        .map(|i| {
            let r = 0.999 * rng.next_f64().sqrt();
            let a = std::f64::consts::TAU * rng.next_f64();
            let x = C32::new((r * a.cos()) as f32, (r * a.sin()) as f32);
            match (specials, i % 97) {
                (_, 0) => C32::new(0.0, -0.0),
                (true, 1) => C32::new(1.0, -1.0),
                (true, 2) => C32::new(f32::NAN, 0.5),
                (true, 3) => C32::new(f32::INFINITY, f32::NEG_INFINITY),
                (true, 4) => C32::new(1e-40, -1e-41),
                _ => x,
            }
        })
        .collect()
}

/// A `CQ15` signal: random components and the extremes −32768 and 32767.
fn cq15_signal(n: usize, seed: u64, full_scale: bool) -> Vec<CQ15> {
    let mut rng = Pcg32::new(seed, 5);
    (0..n)
        .map(|i| {
            let re = (rng.next_u32() >> 16) as u16 as i16;
            let im = (rng.next_u32() >> 16) as u16 as i16;
            let (re, im) = match (full_scale, i % 89) {
                (true, 0) => (i16::MIN, i16::MIN),
                (true, 1) => (i16::MAX, i16::MIN),
                (_, 2) => (0, 0),
                _ => (re, im),
            };
            CQ15 {
                re: Q15(re),
                im: Q15(im),
            }
        })
        .collect()
}

/// The block on the whole buffer and in 20 chunk plans equals `want`, bit for bit.
fn check_plans<T>(c: &ScrambleConfig, input: &[T], want: &[T], what: &str)
where
    T: sdrk_blocks::BlockSample + bytemuck::Pod + core::fmt::Debug,
    Sdrk: Scramble<T>,
{
    let r = rate(48_000);
    let make = || ScrambleBlock::<T, VecStorage>::new(c, r, &mut VecStorage).unwrap();
    let whole = run_block(&mut make(), input, None);
    assert_eq!(bytes(&whole), bytes(want), "{what} whole");
    for plan in twenty_plans(SEED) {
        let got = run_block(&mut make(), input, Some(plan));
        assert_eq!(bytes(&got), bytes(want), "{what} {plan:?}");
    }
}

#[test]
fn the_block_equals_the_kernel_with_the_prbs_indices() {
    for (i, key) in KEYS.into_iter().enumerate() {
        for bits in 1..=8 {
            for inverse in [false, true] {
                let seed = 1 + (i as u32 * 37 + bits * 11) % 31;
                let c = cfg(bits, inverse, key, seed);
                let what = format!("{key:?} bits {bits} inverse {inverse} seed {seed}");
                let x = c32_signal(N, SEED + u64::from(bits), true);
                check_plans(&c, &x, &kernel(&c, &x), &format!("C32 {what}"));
                let x = cq15_signal(N, SEED + u64::from(bits), true);
                check_plans(&c, &x, &kernel(&c, &x), &format!("CQ15 {what}"));
            }
        }
    }
}

/// The phase indices that the block used, read back from its output for the probe input
/// `1 + 0j`: the output of each sample is the phasor of its index (the conjugate for
/// `inverse`), whose angle gives the index (the nearest of the `2^bits` angles, which are
/// 1.4 degrees apart at `bits = 8`; the phasor is within 1e-6 of the exact one, asserted).
fn indices_of_the_block(c: &ScrambleConfig, out: &[C32]) -> Vec<u8> {
    let m = 1u32 << c.bits;
    let step = std::f64::consts::TAU / f64::from(m);
    out.iter()
        .map(|y| {
            let (re, im) = (f64::from(y.re), f64::from(y.im));
            let a = im.atan2(re);
            let k = (a / step).round() as i64;
            let err = (a - k as f64 * step).abs();
            assert!(
                err < 1e-6 && ((re * re + im * im).sqrt() - 1.0).abs() < 1e-6,
                "{c:?}: {y:?} is no phasor"
            );
            let k = if c.inverse { -k } else { k };
            k.rem_euclid(i64::from(m)) as u8
        })
        .collect()
}

#[test]
fn the_index_production_equals_next_bit_over_the_full_period() {
    // M49.1 test 4: every key, every bits, more than one full period of the key (the indices
    // repeat after `period` samples at the latest), a random seed and both directions.
    let mut rng = Pcg32::new(SEED_M49_1, 7);
    let r = rate(48_000);
    for key in KEYS {
        let period = key.period() as usize;
        let n = period + 64;
        let probe = vec![C32::new(1.0, 0.0); n];
        for bits in 1..=8 {
            let seed = 1 + rng.below(key.period());
            let c = cfg(bits, bits % 2 == 0, key, seed);
            let want = indices(key, seed, bits, n);
            let make = || ScrambleBlock::<C32, VecStorage>::new(&c, r, &mut VecStorage).unwrap();
            let got = indices_of_the_block(&c, &run_block(&mut make(), &probe, None));
            assert_eq!(got, want, "{c:?} whole");
            for plan in twenty_plans(SEED_M49_1) {
                let got = indices_of_the_block(&c, &run_block(&mut make(), &probe, Some(plan)));
                assert_eq!(got, want, "{c:?} {plan:?}");
            }
        }
    }
}

#[test]
fn the_default_key_and_the_largest_seed() {
    // prbs15 from 32767 (all ones) at bits 3; the seed of the defaults at bits 1.
    for c in [
        cfg(3, false, Poly::Prbs15, 32767),
        ScrambleConfig::default(),
    ] {
        let x = c32_signal(N, SEED + 1, false);
        check_plans(&c, &x, &kernel(&c, &x), &format!("{c:?}"));
    }
}

#[test]
fn the_indices_are_msb_first_and_run_on_across_calls() {
    // prbs9 from 0x1ff starts 0 0 0 0 0 1 1 1 1 0 1 1 (M38): at bits 3 the first four
    // indices are 000, 001, 111, 011 = 0, 1, 7, 3; at bits 2 the phasors 1, 1, j, -j, ...
    let c = cfg(2, false, Poly::Prbs9, 0x1ff);
    let mut b = ScrambleBlock::<C32, VecStorage>::new(&c, rate(8_000), &mut VecStorage).unwrap();
    let one = [C32::new(1.0, 0.0); 3];
    let mut y = [C32::default(); 3];
    b.process(&one, &mut y).unwrap();
    // 00, 00, 01 -> 1, 1, j
    assert_eq!(
        y,
        [C32::new(1.0, 0.0), C32::new(1.0, 0.0), C32::new(-0.0, 1.0)]
    );
    let mut y = [C32::default(); 3];
    b.process(&one, &mut y).unwrap();
    // 11, 10, 11 -> -j, -1, -j
    assert_eq!(y[0].to_bits_pair(), C32::new(0.0, -1.0).to_bits_pair());
    assert_eq!(y[1].re, -1.0);
    assert_eq!(y[2].im, -1.0);
    assert_eq!(indices(Poly::Prbs9, 0x1ff, 3, 4), [0, 1, 7, 3]);
}

/// The bits of a `C32` as a pair, for comparisons that tell `-0.0` from `+0.0`.
trait Bits {
    fn to_bits_pair(self) -> (u32, u32);
}

impl Bits for C32 {
    fn to_bits_pair(self) -> (u32, u32) {
        (self.re.to_bits(), self.im.to_bits())
    }
}

/// Scrambler then descrambler with the same configuration, in pieces.
fn round_trip<T>(c: &ScrambleConfig, x: &[T], seed: u64) -> Vec<T>
where
    T: sdrk_blocks::BlockSample + Copy + Default,
    Sdrk: Scramble<T>,
{
    let r = rate(48_000);
    let mut tx = ScrambleBlock::<T, VecStorage>::new(c, r, &mut VecStorage).unwrap();
    let rx_cfg = ScrambleConfig {
        inverse: !c.inverse,
        ..*c
    };
    let mut rx = ScrambleBlock::<T, VecStorage>::new(&rx_cfg, r, &mut VecStorage).unwrap();
    let y = run_block(&mut tx, x, Some(sdrk_testkit::ChunkPlan::new(seed, 300)));
    run_block(
        &mut rx,
        &y,
        Some(sdrk_testkit::ChunkPlan::new(seed + 1, 77)),
    )
}

#[test]
fn scramble_then_descramble_returns_the_input() {
    let seed = SEED + 1; // 697
    for key in KEYS {
        for bits in 1..=8 {
            let c = cfg(bits, false, key, 5);
            // C32: exact at bits 1 and 2 (the specials included), SNR >= 120 dB above.
            let x = c32_signal(N, seed, bits <= 2);
            let z = round_trip(&c, &x, seed);
            if bits <= 2 {
                assert_eq!(bytes(&z), bytes(&x), "C32 {key:?} bits {bits}");
            } else {
                let (mut sig, mut err) = (0.0f64, 0.0f64);
                for (a, b) in x.iter().zip(&z) {
                    sig += f64::from(a.re).powi(2) + f64::from(a.im).powi(2);
                    err += f64::from(a.re - b.re).powi(2) + f64::from(a.im - b.im).powi(2);
                }
                let snr = 10.0 * (sig / err).log10();
                assert!(snr >= 120.0, "C32 {key:?} bits {bits}: {snr} dB");
            }
            // CQ15: exact except −32768 at bits 1 and 2; within 2 LSB for |x| <= 1 above.
            let x = if bits <= 2 {
                cq15_signal(N, seed, true)
            } else {
                // The unit disk of the round-trip bound.
                c32_signal(N, seed, false)
                    .iter()
                    .map(|v| CQ15 {
                        re: Q15((v.re * 32767.0).round() as i16),
                        im: Q15((v.im * 32767.0).round() as i16),
                    })
                    .collect()
            };
            let z = round_trip(&c, &x, seed);
            for (i, (a, b)) in x.iter().zip(&z).enumerate() {
                for (p, q) in [(a.re.0, b.re.0), (a.im.0, b.im.0)] {
                    if bits <= 2 {
                        // −32768 may come back as −32767 (the saturating negation), else exact.
                        assert!(
                            q == p || (p == i16::MIN && q == -32767),
                            "CQ15 {key:?} bits {bits} sample {i}: {p} -> {q}"
                        );
                    } else {
                        assert!(
                            (i32::from(p) - i32::from(q)).abs() <= 2,
                            "CQ15 {key:?} bits {bits} sample {i}: {p} -> {q}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn a_call_takes_at_most_scramble_chunk_samples() {
    assert_eq!(SCRAMBLE_CHUNK, 256);
    let c = cfg(5, false, Poly::Prbs11, 77);
    let x = c32_signal(1000, SEED, false);
    let mut b = ScrambleBlock::<C32, VecStorage>::new(&c, rate(8_000), &mut VecStorage).unwrap();
    let mut out = vec![C32::default(); 1000];
    let p = b.process(&x, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (256, 256));
    // The output room limits a call too.
    let p = b.process(&x[256..], &mut out[256..300]).unwrap();
    assert_eq!((p.consumed, p.produced), (44, 44));
    let p = b.process(&x[300..], &mut []).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    let p = b.process(&[], &mut out[300..]).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    let want = kernel(&c, &x[..300]);
    assert_eq!(bytes(&out[..300]), bytes(&want));
}

#[test]
fn reset_restarts_the_key() {
    let c = cfg(4, true, Poly::Prbs7, 100);
    let x = cq15_signal(500, SEED, true);
    let mut b = ScrambleBlock::<CQ15, VecStorage>::new(&c, rate(8_000), &mut VecStorage).unwrap();
    let first = run_block(&mut b, &x, None);
    let _ = run_block(&mut b, &x[..123], None);
    b.reset();
    let again = run_block(&mut b, &x, None);
    assert_eq!(first, again);
    assert_eq!(first, kernel(&c, &x));
}

fn out_of_range(e: Error, name: &str) {
    match e {
        Error::OutOfRange { name: n, .. } => assert_eq!(n, name),
        other => panic!("{name}: {other:?}"),
    }
}

#[test]
fn invalid_parameters_are_errors_at_their_parameter() {
    let r = rate(48_000);
    let new = |c: &ScrambleConfig| ScrambleBlock::<C32, VecStorage>::new(c, r, &mut VecStorage);
    let ok = ScrambleConfig::default();
    for bits in [0, 9, 100] {
        let c = ScrambleConfig { bits, ..ok };
        out_of_range(new(&c).unwrap_err(), "bits");
        out_of_range(c.validate(r).unwrap_err(), "bits");
        out_of_range(
            ScrambleBlock::<CQ15, VecStorage>::storage_bytes(&c, r).unwrap_err(),
            "bits",
        );
    }
    for seed in [0, 32768] {
        out_of_range(new(&ScrambleConfig { seed, ..ok }).unwrap_err(), "seed");
    }
    // A seed in range whose low n bits are all zero.
    for (key, seed) in [
        (Poly::Prbs5, 32),
        (Poly::Prbs7, 128),
        (Poly::Prbs9, 512),
        (Poly::Prbs10, 1024),
        (Poly::Prbs11, 2048 * 3),
    ] {
        let c = ScrambleConfig { key, seed, ..ok };
        for e in [
            new(&c).unwrap_err(),
            c.validate(r).unwrap_err(),
            ScrambleBlock::<C32, VecStorage>::rate_out(&c, r).unwrap_err(),
        ] {
            match e {
                Error::InvalidParameter { name, .. } => assert_eq!(name, "seed"),
                other => panic!("{key:?} {seed}: {other:?}"),
            }
        }
        // The same seed is valid for a longer key.
        assert!(
            ScrambleConfig {
                key: Poly::Prbs15,
                ..c
            }
            .validate(r)
            .is_ok()
        );
    }
    // Through `set`: range and type at the parameter; every key by its choice index.
    let mut c = ok;
    out_of_range(c.set("bits", ParamValue::I64(0)).unwrap_err(), "bits");
    out_of_range(c.set("bits", ParamValue::I64(9)).unwrap_err(), "bits");
    out_of_range(c.set("seed", ParamValue::I64(0)).unwrap_err(), "seed");
    out_of_range(c.set("key", ParamValue::Choice(6)).unwrap_err(), "key");
    // The old name `key_seed` (renamed to `seed` in M50, D2618) has no alias.
    assert_eq!(
        c.set("key_seed", ParamValue::I64(1)),
        Err(Error::UnknownParameter)
    );
    assert_eq!(c.get("key_seed"), Err(Error::UnknownParameter));
    assert_eq!(
        c.set("inverse", ParamValue::I64(1)),
        Err(Error::WrongParamType)
    );
    for (i, key) in KEYS.into_iter().enumerate() {
        c.set("key", ParamValue::Choice(i as u16)).unwrap();
        assert_eq!(c.key, key);
        assert_eq!(c.get("key"), Ok(ParamValue::Choice(i as u16)));
    }
    assert_eq!(
        c.set("nope", ParamValue::Bool(true)),
        Err(Error::UnknownParameter)
    );
    // Every parameter is build-only on the block.
    let mut b = new(&ok).unwrap();
    for name in ["bits", "inverse", "key", "seed"] {
        let v = b.param(name).unwrap();
        assert_eq!(
            b.set_param(name, v),
            Err(Error::ReadOnlyParameter),
            "{name}"
        );
    }
    assert_eq!(
        b.set_param("nope", ParamValue::Bool(true)),
        Err(Error::UnknownParameter)
    );
}

#[test]
fn the_description_follows_the_rules() {
    let d = all_descriptions()
        .iter()
        .find(|d| d.kind == "scramble")
        .unwrap();
    assert!(d.summary.chars().count() <= 120 && d.summary.ends_with('.'));
    assert!(d.details.len() > d.summary.len() && !d.details.contains('\n'));
    for text in [
        "most significant",
        "not secrecy",
        "reset",
        "same sample",
        "DVB-S2",
    ] {
        assert!(d.details.contains(text), "details lack {text}");
    }
    let complex = [PortType::C32, PortType::CQ15];
    assert_eq!(d.inputs.len(), 1);
    assert_eq!(d.inputs[0].types, complex);
    assert_eq!(d.outputs.len(), 1);
    assert_eq!(d.outputs[0].types, complex);
    assert_eq!(d.required.len(), 0);
    let names: Vec<_> = d.params.iter().map(|p| p.name).collect();
    assert_eq!(names, ["bits", "inverse", "key", "seed"]);
    for p in d.params {
        assert_eq!(p.unit, Unit::None, "{}", p.name);
        assert_eq!(p.mutability, Mutability::Build, "{}", p.name);
        assert!(!p.typical.is_empty() && !p.description.is_empty());
    }
    assert_eq!(
        d.param("bits").unwrap().kind,
        ParamKind::I64 {
            min: 1,
            max: 8,
            default: Some(1)
        }
    );
    assert_eq!(
        d.param("inverse").unwrap().kind,
        ParamKind::Bool { default: false }
    );
    assert_eq!(
        d.param("key").unwrap().kind,
        ParamKind::Choice {
            options: &["prbs5", "prbs7", "prbs9", "prbs10", "prbs11", "prbs15"],
            default: 5
        }
    );
    assert_eq!(
        d.param("seed").unwrap().kind,
        ParamKind::I64 {
            min: 1,
            max: 32767,
            default: Some(1)
        }
    );
    let c = ScrambleConfig::default();
    assert_eq!(
        (c.bits, c.inverse, c.key, c.seed),
        (1, false, Poly::Prbs15, 1)
    );
}

#[test]
fn the_block_needs_no_storage_and_keeps_the_rate() {
    let c = cfg(6, false, Poly::Prbs10, 3);
    let r = rate(250_000);
    assert_eq!(
        ScrambleBlock::<C32, VecStorage>::storage_bytes(&c, r),
        Ok(0)
    );
    assert_eq!(
        ScrambleBlock::<CQ15, VecStorage>::storage_bytes(&c, r),
        Ok(0)
    );
    assert_eq!(ScrambleBlock::<C32, VecStorage>::rate_out(&c, r), Ok(r));
    check_exact_storage(0, |a| ScrambleBlock::<C32, _>::new(&c, r, a).map(|_| ()));
    check_exact_storage(0, |a| ScrambleBlock::<CQ15, _>::new(&c, r, a).map(|_| ()));
}

fn check_timing<B: Block>(b: &B) {
    let t = b.timing();
    assert_eq!(t.ratio, Ratio::ONE);
    assert_eq!(t.kind, RatioKind::Constant);
    assert_eq!(t.delay, Delay::ZERO);
    assert_eq!(b.min_input(), 1);
    assert_eq!(b.description().kind, "scramble");
}

#[test]
fn timing_is_ratio_one_and_delay_zero() {
    let c = cfg(8, true, Poly::Prbs5, 9);
    let r = rate(48_000);
    let mut arena = ArenaStorage::new(&mut []);
    check_timing(&ScrambleBlock::<C32, _>::new(&c, r, &mut arena).unwrap());
    let mut b = ScrambleBlock::<CQ15, _>::new(&c, r, &mut arena).unwrap();
    b.process(&[CQ15::default(); 9], &mut [CQ15::default(); 9])
        .unwrap();
    check_timing(&b);
}

#[test]
fn the_block_is_small_send_and_sync() {
    fn check<T: Send + Sync>(name: &str) {
        let size = std::mem::size_of::<T>();
        assert!(size <= 512, "{name} is {size} bytes");
    }
    check::<ScrambleBlock<C32, ArenaStorage<'static>>>("ScrambleBlock<C32>");
    check::<ScrambleBlock<CQ15, VecStorage>>("ScrambleBlock<CQ15, Vec>");
    check::<ScrambleConfig>("ScrambleConfig");
}
