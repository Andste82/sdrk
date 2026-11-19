//! The blocks `bit_scramble` and `bit_descramble` (M44.2, the maintainer's decision on I44.2-6):
//! the self-synchronizing scrambler of a soft-bit stream and its descrambler, for every
//! polynomial and constellation.
//!
//! - Round trip (seed 838): `bit_descramble` after `bit_scramble` with the same configuration
//!   gives the input bit for bit (random soft bits of random magnitude, none 0).
//! - Self-synchronization (seed 839): a descrambler that joins the scrambled stream at any bit,
//!   from its zero register, gives the input from `A = a m` bits after its start on (`a` the
//!   degree of the polynomial, `m` the bits per symbol), and its first output bits are not all
//!   right (it needs the register).
//! - Error propagation (seed 873, stream 2): one wrong bit on the air gives exactly 3 wrong bits out, at
//!   the bit and `B = b m` and `A` bits later, as the polynomial `1 + x^-b + x^-a` predicts; the
//!   magnitudes stay.
//! - Chunk invariance (seed 872): both blocks give the bits of one call in 20 cut plans, and
//!   `process_in_place` the bits of `process`.
//! - Quarter turns (seed 873, stream 1): for every square constellation, a rotation of the scrambled
//!   symbols by 90 degrees (Gray code per axis, `sdrk_proto::frame::rotate_group`) descrambles to
//!   the same rotation of the plain symbols, after the register has filled.
//! - Construction (review of M44.2, D1536): both blocks build in an arena of exactly
//!   `storage_bytes` for every polynomial and constellation (and fail with one byte less), and
//!   the public types are `Send` and `Sync` (R4.6).
//! - The model: the output equals a one-bit-at-a-time model of the register (soft bits: the sign
//!   flips where the parity of the register is 1), including 0 and NaN inputs (seed 838). The
//!   golden cases of the Python model are `blocks/bit_scramble_*` (seeds 870 and 871,
//!   `sdrk-testkit` `golden_m44_2`).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::trivially_copy_pass_by_ref,
    clippy::items_after_statements
)]

mod common;

use common::{check_block, check_exact_storage, rate, run_block, same_bits};
use sdrk_blocks::{
    ArenaStorage, BitDescrambleBlock, BitPolynomial, BitScrambleBlock, BitScrambleConfig, Block,
    ParamValue, Scrambler, SymbolMapConstellation,
};
use sdrk_proto::frame::rotate_group;
use sdrk_testkit::{Pcg32, VecStorage};

const POLYNOMIALS: [BitPolynomial; 4] = [
    BitPolynomial::X23X18,
    BitPolynomial::X23X5,
    BitPolynomial::X17X12,
    BitPolynomial::X58X39,
];

const CONSTELLATIONS: [SymbolMapConstellation; 7] = [
    SymbolMapConstellation::Bpsk,
    SymbolMapConstellation::Qpsk,
    SymbolMapConstellation::Psk8,
    SymbolMapConstellation::Qam16,
    SymbolMapConstellation::Qam64,
    SymbolMapConstellation::Qam256,
    SymbolMapConstellation::Qam1024,
];

/// Every configuration.
fn configs() -> Vec<BitScrambleConfig> {
    POLYNOMIALS
        .iter()
        .flat_map(|&polynomial| {
            CONSTELLATIONS
                .iter()
                .map(move |&constellation| BitScrambleConfig {
                    polynomial,
                    constellation,
                })
        })
        .collect()
}

fn scrambler(cfg: &BitScrambleConfig) -> BitScrambleBlock<VecStorage> {
    BitScrambleBlock::new(cfg, rate(9600), &mut VecStorage).unwrap()
}

fn descrambler(cfg: &BitScrambleConfig) -> BitDescrambleBlock<VecStorage> {
    BitDescrambleBlock::new(cfg, rate(9600), &mut VecStorage).unwrap()
}

/// `n` soft bits of random sign and magnitude (0.01 to 4), none 0.
fn soft_bits(rng: &mut Pcg32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let m = 0.01 + 3.99 * rng.next_f64();
            let v = if rng.below(2) == 1 { m } else { -m };
            v as f32
        })
        .collect()
}

fn scramble(cfg: &BitScrambleConfig, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    let p = scrambler(cfg).process(x, &mut y).unwrap();
    assert_eq!((p.consumed, p.produced), (x.len(), x.len()));
    y
}

fn descramble(cfg: &BitScrambleConfig, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    let p = descrambler(cfg).process(x, &mut y).unwrap();
    assert_eq!((p.consumed, p.produced), (x.len(), x.len()));
    y
}

fn hard(x: &[f32]) -> Vec<bool> {
    x.iter().map(|&v| v > 0.0).collect()
}

/// The one-bit-at-a-time model: `descramble` false scrambles (the register holds the output
/// bits), true descrambles (the register holds the input bits); the output is the input with
/// its sign flipped where `r[n - a] ^ r[n - b]` is 1.
fn model(cfg: &BitScrambleConfig, x: &[f32], descramble: bool) -> Vec<f32> {
    let (a, b) = cfg.delays();
    let mut reg: Vec<bool> = Vec::with_capacity(x.len());
    let at = |reg: &Vec<bool>, n: usize, d: usize| n >= d && reg[n - d];
    x.iter()
        .enumerate()
        .map(|(n, &v)| {
            let p = at(&reg, n, a) ^ at(&reg, n, b);
            let y = if p {
                f32::from_bits(v.to_bits() ^ 0x8000_0000)
            } else {
                v
            };
            reg.push(if descramble { v > 0.0 } else { y > 0.0 });
            y
        })
        .collect()
}

#[test]
fn the_delays_count_symbols() {
    for cfg in configs() {
        let (a, b) = cfg.polynomial.taps();
        let m = cfg.constellation.bits_per_symbol();
        assert_eq!(cfg.delays(), (a * m, b * m), "{cfg:?}");
    }
    assert_eq!(BitScrambleConfig::default().delays(), (23, 18));
}

#[test]
fn both_blocks_follow_the_bit_model() {
    let mut rng = Pcg32::new(838, 1);
    for cfg in configs() {
        let mut x = soft_bits(&mut rng, 3000);
        // A 0, a negative 0 and a NaN count as 0 bits and keep their value up to the sign.
        x[100] = 0.0;
        x[700] = -0.0;
        x[1500] = f32::NAN;
        let s = scramble(&cfg, &x);
        let want = model(&cfg, &x, false);
        for (i, (g, w)) in s.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "scramble {cfg:?} at {i}");
        }
        let d = descramble(&cfg, &s);
        let want = model(&cfg, &s, true);
        for (i, (g, w)) in d.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "descramble {cfg:?} at {i}");
        }
    }
}

#[test]
fn descramble_after_scramble_gives_the_input() {
    let mut rng = Pcg32::new(838, 2);
    for cfg in configs() {
        let x = soft_bits(&mut rng, 5000);
        let air = scramble(&cfg, &x);
        // The magnitudes stay, and the bits on the air are not the bits sent.
        assert!(air.iter().zip(&x).all(|(a, x)| a.abs() == x.abs()));
        assert_ne!(hard(&air), hard(&x), "{cfg:?}");
        let back = descramble(&cfg, &air);
        same_bits(&x, &back, &format!("{cfg:?}"));
    }
}

#[test]
fn a_constant_stream_becomes_balanced() {
    // A run of ones (or zeros) scrambles to a stream with about as many ones as zeros, the
    // purpose of the block; the ones come from the register filled by the first output bits.
    for cfg in configs() {
        for v in [1.0f32, -1.0] {
            let air = scramble(&cfg, &vec![v; 20_000]);
            let ones = air[2000..].iter().filter(|&&x| x > 0.0).count();
            let share = ones as f64 / 18_000.0;
            assert!(
                (0.4..0.6).contains(&share) || (v < 0.0 && ones == 0),
                "{cfg:?} {v}: {share}"
            );
        }
    }
}

#[test]
fn the_descrambler_locks_after_the_register_length_from_any_start() {
    let mut rng = Pcg32::new(839, 1);
    for cfg in configs() {
        let (a, _) = cfg.delays();
        let x = soft_bits(&mut rng, 4000 + a);
        let air = scramble(&cfg, &x);
        let mut wrong_before = 0;
        for _ in 0..8 {
            let start = 1 + rng.below(3000) as usize;
            let out = descramble(&cfg, &air[start..]);
            same_bits(
                &x[start + a..],
                &out[a..],
                &format!("{cfg:?} from bit {start}"),
            );
            wrong_before += hard(&out[..a])
                .iter()
                .zip(hard(&x[start..start + a]))
                .filter(|(o, x)| **o != *x)
                .count();
        }
        assert!(wrong_before > 0, "{cfg:?}: the register does not matter");
    }
}

#[test]
fn one_wrong_bit_on_the_air_gives_three_wrong_bits() {
    let mut rng = Pcg32::new(873, 2);
    for cfg in configs() {
        let (a, b) = cfg.delays();
        let x = soft_bits(&mut rng, 2000 + a);
        let air = scramble(&cfg, &x);
        for _ in 0..4 {
            let e = rng.below(1500) as usize;
            let mut bad = air.clone();
            bad[e] = -bad[e];
            let out = descramble(&cfg, &bad);
            let wrong: Vec<usize> = (0..x.len()).filter(|&i| out[i] != x[i]).collect();
            assert_eq!(wrong, [e, e + b, e + a], "{cfg:?}, error at {e}");
            // A wrong bit flips the sign; the magnitude (the reliability) stays.
            assert!(out.iter().zip(&x).all(|(o, x)| o.abs() == x.abs()));
        }
    }
}

#[test]
fn both_blocks_do_not_depend_on_the_cut_plan() {
    let mut rng = Pcg32::new(872, 1);
    for (k, cfg) in configs().into_iter().enumerate() {
        let x = soft_bits(&mut rng, 6000);
        let want = scramble(&cfg, &x);
        let seed = 872 * 100 + k as u64;
        check_block(
            &want,
            &x,
            seed,
            || scrambler(&cfg),
            &format!("scramble {cfg:?}"),
        );
        let back = descramble(&cfg, &want);
        check_block(
            &back,
            &want,
            seed,
            || descrambler(&cfg),
            &format!("descramble {cfg:?}"),
        );
        // In place, in pieces of every length from 1 to 130.
        let mut s = scrambler(&cfg);
        let mut d = descrambler(&cfg);
        let (mut y, mut z) = (x.clone(), want.clone());
        let mut at = 0;
        let mut len = 1;
        while at < x.len() {
            let n = len.min(x.len() - at);
            s.process_in_place(&mut y[at..at + n]);
            d.process_in_place(&mut z[at..at + n]);
            at += n;
            len = len % 130 + 1;
        }
        same_bits(&want, &y, &format!("scramble in place {cfg:?}"));
        same_bits(&back, &z, &format!("descramble in place {cfg:?}"));
    }
}

#[test]
fn a_reset_starts_from_the_zero_register() {
    let mut rng = Pcg32::new(872, 2);
    let cfg = BitScrambleConfig {
        constellation: SymbolMapConstellation::Qam256,
        ..BitScrambleConfig::default()
    };
    let x = soft_bits(&mut rng, 1000);
    let mut s = scrambler(&cfg);
    let first = run_block(&mut s, &x, None);
    s.reset();
    same_bits(&first, &run_block(&mut s, &x, None), "scramble after reset");
    let mut d = descrambler(&cfg);
    let back = run_block(&mut d, &first, None);
    d.reset();
    same_bits(
        &back,
        &run_block(&mut d, &first, None),
        "descramble after reset",
    );
    same_bits(&x, &back, "round trip");
}

/// The bits of the symbol `k` (`m` bits, the first the most significant) as a number.
fn group(bits: &[bool], k: usize, m: usize) -> u32 {
    bits[k * m..(k + 1) * m]
        .iter()
        .fold(0, |g, &b| (g << 1) | u32::from(b))
}

/// The soft bits `x` with every symbol of `m` bits turned by `q` quarter turns (the bits that
/// the rotation inverts flip their sign, the others move; magnitudes move with their bits).
fn rotate(x: &[f32], m: usize, q: u32) -> Vec<f32> {
    let bits = hard(x);
    let mut out = vec![0.0f32; x.len()];
    for k in 0..x.len() / m {
        // Where every bit of the symbol goes: the rotation of the one-hot groups (an affine map
        // over GF(2): the image of 0 is the constant, a one-hot image minus it the moved bit).
        let base = rotate_group(0, (m / 2) as u32, q);
        let g = rotate_group(group(&bits, k, m), (m / 2) as u32, q);
        for i in 0..m {
            let one = rotate_group(1 << (m - 1 - i), (m / 2) as u32, q) ^ base;
            let to = (0..m).find(|&j| one >> (m - 1 - j) & 1 == 1).unwrap();
            let v = x[k * m + i].abs();
            out[k * m + to] = if g >> (m - 1 - to) & 1 == 1 { v } else { -v };
        }
    }
    out
}

#[test]
fn a_quarter_turn_commutes_with_the_descrambler() {
    let mut rng = Pcg32::new(873, 1);
    for cfg in configs() {
        let m = cfg.constellation.bits_per_symbol();
        if m % 2 == 1 {
            continue; // bpsk and 8psk are not square
        }
        let (a, _) = cfg.delays();
        let x = soft_bits(&mut rng, 600 * m + a);
        let air = scramble(&cfg, &x);
        for q in 0..4 {
            let turned = descramble(&cfg, &rotate(&air, m, q));
            let want = rotate(&x, m, q);
            // After the register has filled (its first `a` symbols hold the zero register,
            // which the rotation does not turn).
            assert_eq!(
                hard(&turned[a..]),
                hard(&want[a..]),
                "{cfg:?}, {q} quarter turns"
            );
        }
    }
}

#[test]
fn the_parameters_are_build_parameters() {
    let cfg = BitScrambleConfig {
        polynomial: BitPolynomial::X58X39,
        constellation: SymbolMapConstellation::Qam64,
    };
    let mut s = scrambler(&cfg);
    let mut d = descrambler(&cfg);
    assert_eq!(s.param("polynomial"), Ok(ParamValue::Choice(3)));
    assert_eq!(d.param("constellation"), Ok(ParamValue::Choice(4)));
    fn read_only<B: Block>(b: &mut B) {
        assert_eq!(
            b.set_param("polynomial", ParamValue::Choice(0)),
            Err(sdrk_blocks::Error::ReadOnlyParameter)
        );
        assert_eq!(b.param("taps"), Err(sdrk_blocks::Error::UnknownParameter));
    }
    read_only(&mut s);
    read_only(&mut d);
    let mut c = BitScrambleConfig::default();
    assert!(c.set("polynomial", ParamValue::Choice(4)).is_err());
    assert!(c.set("constellation", ParamValue::Choice(7)).is_err());
    c.set("polynomial", ParamValue::Choice(1)).unwrap();
    assert_eq!(c.polynomial, BitPolynomial::X23X5);
    assert_eq!(s.config(), cfg);
    assert_eq!(d.config(), cfg);
}

#[test]
fn both_blocks_take_exactly_their_storage() {
    for cfg in configs() {
        let r = rate(9600);
        let bytes = BitScrambleBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, r).unwrap();
        // The ring of `A = a m` soft bits (at most 580: x58_x39 over 1024-QAM).
        assert!(cfg.delays().0 <= 580, "{cfg:?}");
        assert_eq!(bytes, cfg.delays().0 * 4, "{cfg:?}");
        check_exact_storage(bytes, |a| BitScrambleBlock::new(&cfg, r, a).map(|_| ()));
        assert_eq!(
            BitDescrambleBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, r).unwrap(),
            bytes,
            "{cfg:?}"
        );
        check_exact_storage(bytes, |a| BitDescrambleBlock::new(&cfg, r, a).map(|_| ()));
    }
}

#[test]
fn the_public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BitScrambleBlock<ArenaStorage<'static>>>();
    assert_send_sync::<BitScrambleBlock<VecStorage>>();
    assert_send_sync::<BitDescrambleBlock<ArenaStorage<'static>>>();
    assert_send_sync::<BitDescrambleBlock<VecStorage>>();
    assert_send_sync::<BitScrambleConfig>();
    assert_send_sync::<BitPolynomial>();
    assert_send_sync::<Scrambler>();
}
