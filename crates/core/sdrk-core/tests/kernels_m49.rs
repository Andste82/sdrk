//! Tests of the M49 kernel `scramble` and the exact `±1`/`±j` helpers of `sdrk_core::num`
//! (milestone M49, tests 2 to 4): parameters and the index mask, the helpers of §1.1 by brute
//! force over every `Q15` value, `k = 0` as the identity, `bits` 1 and 2 as exact quadrant maps
//! and their round trips, the direct table lookup against the `rotate` interpolation (through the
//! public `Rotate` kernel), the residual `rotate` step plus the quadrant map, the `inverse` index
//! and the round-trip bounds at `bits` 3 to 8, `Fallback` against `Reference` with special
//! inputs, the per-sample helpers, block-size invariance (seed 695) and the extremes.

#![cfg(feature = "reference")]
#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::print_stderr
)]

use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::kernels::scramble::{
    Scramble, phase_index, residual_c32, residual_q15, scramble_one_c32, scramble_one_cq15,
    split_index,
};
use sdrk_core::num::{c32_chip, c32_quadrant, cq15_chip, cq15_quadrant, q15_neg_sat};
use sdrk_core::tables::{COS_F32, COS_Q15};
use sdrk_core::{C32, CQ15, Error, Fallback, Q15, Reference};

/// Seed of the random inputs of the unit tests (M49 seeds 690 to 699).
const SEED: u64 = 690;
/// Seed of the chunk plans (test 3).
const PLAN_SEED: u64 = 695;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }

    /// A uniform value in `[-1, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    fn q15(&mut self) -> Q15 {
        match self.below(8) {
            0 => Q15::MIN,
            1 => Q15::MAX,
            _ => Q15((self.next() >> 20) as i16),
        }
    }

    fn cq15(&mut self) -> CQ15 {
        CQ15 {
            re: self.q15(),
            im: self.q15(),
        }
    }

    fn c32(&mut self) -> C32 {
        C32::new(self.unit() as f32, self.unit() as f32)
    }

    /// A `CQ15` sample of complex magnitude at most 1 (uniform in the disk).
    fn cq15_disk(&mut self) -> CQ15 {
        loop {
            let re = (self.next() >> 48) as i32 - 32768;
            let im = (self.next() >> 48) as i32 - 32768;
            if i64::from(re).pow(2) + i64::from(im).pow(2) <= 1 << 30 {
                return CQ15 {
                    re: Q15(re as i16),
                    im: Q15(im as i16),
                };
            }
        }
    }
}

fn cq(re: i16, im: i16) -> CQ15 {
    CQ15 {
        re: Q15(re),
        im: Q15(im),
    }
}

fn bits_of(x: C32) -> (u32, u32) {
    (x.re.to_bits(), x.im.to_bits())
}

/// The table of §1.1 written out, with the negation `neg`.
fn quadrant_table<T: Copy>(re: T, im: T, q: u32, neg: impl Fn(T) -> T) -> (T, T) {
    match q & 3 {
        0 => (re, im),
        1 => (neg(im), re),
        2 => (neg(re), neg(im)),
        _ => (im, neg(re)),
    }
}

fn f32_neg(x: f32) -> f32 {
    f32::from_bits(x.to_bits() ^ 0x8000_0000)
}

/// Special `f32` values: signed zeros, infinities, subnormals, NaN payloads of both signs.
const SPECIAL_BITS: [u32; 14] = [
    0x0000_0000,
    0x8000_0000,
    0x7f80_0000,
    0xff80_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x807f_ffff,
    0x7fc0_0000,
    0xffc0_0000,
    0x7f80_0001,
    0x7fc1_2345,
    0xffd5_5555,
    0x3f80_0000,
];

fn specials() -> impl Iterator<Item = f32> {
    SPECIAL_BITS.iter().map(|&b| f32::from_bits(b))
}

/// Runs the kernel `k` and returns the output.
fn run_c32<K: Scramble<C32>>(k: &K, x: &[C32], p: &[u8], bits: u32, inverse: bool) -> Vec<C32> {
    let mut y = vec![C32::new(7.0, 7.0); x.len()];
    k.scramble(x, p, bits, inverse, &mut y).unwrap();
    y
}

fn run_cq15<K: Scramble<CQ15>>(k: &K, x: &[CQ15], p: &[u8], bits: u32, inverse: bool) -> Vec<CQ15> {
    let mut y = vec![cq(7, 7); x.len()];
    k.scramble(x, p, bits, inverse, &mut y).unwrap();
    y
}

/// One `rotate` step at the constant phase `phase` (a zero-increment oscillator).
fn rotate_c32(x: C32, phase: u32) -> C32 {
    let mut nco = Nco::from_increment(0);
    nco.set_phase(phase);
    let mut y = [C32::default()];
    Fallback.rotate(&mut nco, &[x], &mut y).unwrap();
    y[0]
}

fn rotate_cq15(x: CQ15, phase: u32) -> CQ15 {
    let mut nco = Nco::from_increment(0);
    nco.set_phase(phase);
    let mut y = [CQ15::default()];
    Fallback.rotate(&mut nco, &[x], &mut y).unwrap();
    y[0]
}

/// SNR in dB of `got` against `want`.
fn snr_db(want: &[C32], got: &[C32]) -> f64 {
    let (mut s, mut e) = (0.0f64, 0.0f64);
    for (a, b) in want.iter().zip(got) {
        let (ar, ai) = (f64::from(a.re), f64::from(a.im));
        let (dr, di) = (ar - f64::from(b.re), ai - f64::from(b.im));
        s += ar * ar + ai * ai;
        e += dr * dr + di * di;
    }
    if e == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (s / e).log10()
    }
}

fn snr_db_q15(want: &[CQ15], got: &[CQ15]) -> f64 {
    let f = |v: &[CQ15]| -> Vec<C32> {
        v.iter()
            .map(|x| C32::new(f32::from(x.re.0), f32::from(x.im.0)))
            .collect()
    };
    snr_db(&f(want), &f(got))
}

// ---------------------------------------------------------------------------------------------
// Parameters and the index mask.

#[test]
fn bits_0_and_9_are_invalid_and_nothing_is_written() {
    let x = [C32::new(1.0, 2.0); 4];
    let xq = [cq(1, 2); 4];
    for bits in [0, 9, 32, u32::MAX] {
        let mut y = [C32::new(5.0, 5.0); 4];
        let mut yq = [cq(5, 5); 4];
        for r in [
            Fallback.scramble(&x, &[1; 4], bits, false, &mut y),
            Reference.scramble(&x, &[1; 4], bits, true, &mut y),
        ] {
            assert!(matches!(r, Err(Error::InvalidParameter(_))), "bits {bits}");
        }
        for r in [
            Fallback.scramble(&xq, &[1; 4], bits, false, &mut yq),
            Reference.scramble(&xq, &[1; 4], bits, true, &mut yq),
        ] {
            assert!(matches!(r, Err(Error::InvalidParameter(_))), "bits {bits}");
        }
        assert_eq!(y, [C32::new(5.0, 5.0); 4]);
        assert_eq!(yq, [cq(5, 5); 4]);
        assert!(matches!(
            scramble_one_c32(x[0], 1, bits),
            Err(Error::InvalidParameter(_))
        ));
        assert!(matches!(
            scramble_one_cq15(xq[0], 1, bits),
            Err(Error::InvalidParameter(_))
        ));
        // The parameter is checked before the lengths.
        assert!(matches!(
            Fallback.scramble(&x, &[1; 3], bits, false, &mut y),
            Err(Error::InvalidParameter(_))
        ));
    }
}

#[test]
fn length_mismatches_are_errors_and_nothing_is_written() {
    let x = [C32::new(1.0, 2.0); 4];
    let xq = [cq(1, 2); 4];
    for (np, ny) in [(3, 4), (5, 4), (4, 3), (4, 5), (0, 0)] {
        let p = vec![1u8; np];
        let mut y = vec![C32::new(5.0, 5.0); ny];
        let mut yq = vec![cq(5, 5); ny];
        for bits in [1, 2, 3, 8] {
            for r in [
                Fallback.scramble(&x, &p, bits, false, &mut y),
                Reference.scramble(&x, &p, bits, false, &mut y),
            ] {
                assert!(matches!(r, Err(Error::LengthMismatch { .. })));
            }
            for r in [
                Fallback.scramble(&xq, &p, bits, true, &mut yq),
                Reference.scramble(&xq, &p, bits, true, &mut yq),
            ] {
                assert!(matches!(r, Err(Error::LengthMismatch { .. })));
            }
        }
        assert!(y.iter().all(|v| *v == C32::new(5.0, 5.0)));
        assert!(yq.iter().all(|v| *v == cq(5, 5)));
    }
}

#[test]
fn every_index_is_taken_modulo_2_pow_bits() {
    let mut rng = Rng::new(SEED);
    let x: Vec<C32> = (0..256).map(|_| rng.c32()).collect();
    let xq: Vec<CQ15> = (0..256).map(|_| rng.cq15()).collect();
    let all: Vec<u8> = (0..=255).collect();
    for bits in 1..=8 {
        let masked: Vec<u8> = all
            .iter()
            .map(|&p| p & ((1u16 << bits) - 1) as u8)
            .collect();
        for inverse in [false, true] {
            assert_eq!(
                run_c32(&Fallback, &x, &all, bits, inverse)
                    .iter()
                    .map(|v| bits_of(*v))
                    .collect::<Vec<_>>(),
                run_c32(&Fallback, &x, &masked, bits, inverse)
                    .iter()
                    .map(|v| bits_of(*v))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                run_cq15(&Fallback, &xq, &all, bits, inverse),
                run_cq15(&Fallback, &xq, &masked, bits, inverse)
            );
        }
    }
    // The example of the spec: p = 255 at bits 2 equals p = 3.
    let x1 = [C32::new(0.25, -0.5)];
    assert_eq!(
        run_c32(&Fallback, &x1, &[255], 2, false),
        run_c32(&Fallback, &x1, &[3], 2, false)
    );
}

#[test]
fn the_index_decomposition_follows_the_contract() {
    for bits in 1..=8u32 {
        for k in 0..(1u32 << bits) {
            let (q, r) = split_index(k, bits);
            if bits == 1 {
                assert_eq!((q, r), (2 * k, 0));
            } else {
                assert_eq!(q, k >> (bits - 2));
                assert_eq!(r, k & ((1 << (bits - 2)) - 1));
                assert!(q < 4);
            }
            // The angle 2π k / 2^b equals q π/2 + 2π r / 2^b.
            let lhs = f64::from(k) / f64::from(1u32 << bits);
            let rhs = f64::from(q) / 4.0 + f64::from(r) / f64::from(1u32 << bits);
            assert_eq!(lhs, rhs);
        }
        for p in 0..=255u8 {
            let k = u32::from(p) & ((1 << bits) - 1);
            assert_eq!(phase_index(p, bits, false), k);
            assert_eq!(
                phase_index(p, bits, true),
                (1u32 << bits).wrapping_sub(k) & ((1 << bits) - 1)
            );
            assert_eq!(
                (phase_index(p, bits, false) + phase_index(p, bits, true)) % (1 << bits),
                0
            );
        }
    }
}

/// R5.4: the public index helpers never panic, also for `bits` outside 1 to 8 (the test build
/// has overflow checks on, so an unchecked `1 << bits` would panic here).
#[test]
fn the_index_helpers_do_not_panic_outside_the_bits_range() {
    for bits in [0, 9, 31, 32, 33, 34, 63, 64, 1000, u32::MAX - 1, u32::MAX] {
        for p in [0u8, 1, 0x80, 0xff] {
            let _ = phase_index(p, bits, false);
            let _ = phase_index(p, bits, true);
        }
        for k in [0, 1, 0xff, u32::MAX] {
            let _ = split_index(k, bits);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The helpers of §1.1.

#[test]
fn q15_neg_sat_negates_every_value_and_saturates_only_minus_one() {
    assert_eq!(q15_neg_sat(Q15(-32768)), Q15(32767));
    for v in i16::MIN + 1..=i16::MAX {
        assert_eq!(q15_neg_sat(Q15(v)), Q15(-v));
    }
}

#[test]
fn the_quadrant_maps_equal_the_table_for_q_0_to_7() {
    let mut rng = Rng::new(SEED + 1);
    for _ in 0..20_000 {
        let x = rng.cq15();
        let c = rng.c32();
        for q in 0..8 {
            let (re, im) = quadrant_table(x.re, x.im, q, q15_neg_sat);
            assert_eq!(cq15_quadrant(x, q), CQ15 { re, im }, "{x:?} q {q}");
            let (re, im) = quadrant_table(c.re, c.im, q, f32_neg);
            assert_eq!(bits_of(c32_quadrant(c, q)), bits_of(C32::new(re, im)));
        }
        for q in [u32::MAX, u32::MAX - 1, 1 << 31] {
            assert_eq!(cq15_quadrant(x, q), cq15_quadrant(x, q & 3));
        }
    }
}

#[test]
fn the_cq15_quadrant_round_trip_is_exact_except_minus_one_in_the_negated_components() {
    for q in 0..4u32 {
        let qi = (4 - q) & 3;
        for v in i16::MIN..=i16::MAX {
            for w in [0i16, 1, -1, 12345, i16::MIN, i16::MAX] {
                for x in [cq(v, w), cq(w, v)] {
                    let back = cq15_quadrant(cq15_quadrant(x, q), qi);
                    let re_neg = q == 2 || q == 3;
                    let im_neg = q == 1 || q == 2;
                    let want_re = if re_neg && x.re.0 == i16::MIN {
                        -32767
                    } else {
                        x.re.0
                    };
                    let want_im = if im_neg && x.im.0 == i16::MIN {
                        -32767
                    } else {
                        x.im.0
                    };
                    assert_eq!(back, cq(want_re, want_im), "x {x:?}, q {q}");
                }
            }
        }
    }
}

#[test]
fn c32_quadrant_is_exact_on_special_values() {
    for re in specials() {
        for im in specials() {
            let x = C32::new(re, im);
            for q in 0..4 {
                let (wr, wi) = quadrant_table(re, im, q, f32_neg);
                let y = c32_quadrant(x, q);
                assert_eq!(bits_of(y), (wr.to_bits(), wi.to_bits()), "{x:?} q {q}");
                // The round trip is exact by bits, NaN payloads included.
                assert_eq!(bits_of(c32_quadrant(y, (4 - q) & 3)), bits_of(x));
                // An infinity stays an infinity, never NaN.
                if re.is_infinite() {
                    assert!(y.re.is_infinite() || y.im.is_infinite());
                }
            }
        }
    }
}

#[test]
fn the_chips_are_the_identity_and_the_negation() {
    let mut rng = Rng::new(SEED + 2);
    for _ in 0..1000 {
        let x = rng.cq15();
        assert_eq!(cq15_chip(x, 0), x);
        assert_eq!(
            cq15_chip(x, 1),
            cq(x.re.0.saturating_neg(), x.im.0.saturating_neg())
        );
        assert_eq!(cq15_chip(x, 2), x);
        assert_eq!(cq15_chip(x, 3), cq15_chip(x, 1));
    }
    assert_eq!(cq15_chip(cq(-32768, -32768), 1), cq(32767, 32767));
    for re in specials() {
        for im in specials() {
            let x = C32::new(re, im);
            assert_eq!(bits_of(c32_chip(x, 0)), bits_of(x));
            assert_eq!(
                bits_of(c32_chip(x, 1)),
                bits_of(C32::new(f32_neg(re), f32_neg(im)))
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// k = 0, bits 1 and 2.

#[test]
fn index_0_is_the_identity_at_every_bits() {
    let mut rng = Rng::new(SEED + 3);
    let mut xq: Vec<CQ15> = (0..1000).map(|_| rng.cq15()).collect();
    xq.extend([
        cq(-32768, -32768),
        cq(32767, 32767),
        cq(-32768, 32767),
        cq(0, 0),
    ]);
    let mut x: Vec<C32> = (0..1000).map(|_| rng.c32()).collect();
    for re in specials() {
        for im in specials() {
            x.push(C32::new(re, im));
        }
    }
    for bits in 1..=8 {
        for inverse in [false, true] {
            // p = 0 and p = 2^bits (masked to 0, except at bits 8 where 256 is not a byte).
            for p in [0u8, (1u16 << bits) as u8] {
                let y = run_c32(&Fallback, &x, &vec![p; x.len()], bits, inverse);
                for (a, b) in x.iter().zip(&y) {
                    assert_eq!(bits_of(*a), bits_of(*b), "bits {bits}");
                }
                let r = run_c32(&Reference, &x, &vec![p; x.len()], bits, inverse);
                for (a, b) in x.iter().zip(&r) {
                    if a.re.is_nan() || a.im.is_nan() {
                        assert_eq!(
                            (a.re.is_nan(), a.im.is_nan()),
                            (b.re.is_nan(), b.im.is_nan())
                        );
                    } else {
                        assert_eq!(bits_of(*a), bits_of(*b));
                    }
                }
                assert_eq!(
                    run_cq15(&Fallback, &xq, &vec![p; xq.len()], bits, inverse),
                    xq
                );
                assert_eq!(
                    run_cq15(&Reference, &xq, &vec![p; xq.len()], bits, inverse),
                    xq
                );
            }
        }
    }
}

#[test]
fn bits_1_and_2_are_the_exact_sign_and_quadrant_maps() {
    let mut rng = Rng::new(SEED + 4);
    let x: Vec<C32> = (0..4000).map(|_| rng.c32()).collect();
    let xq: Vec<CQ15> = (0..4000).map(|_| rng.cq15()).collect();
    let p: Vec<u8> = (0..4000).map(|_| rng.byte()).collect();
    for inverse in [false, true] {
        let y1 = run_c32(&Fallback, &x, &p, 1, inverse);
        let y1q = run_cq15(&Fallback, &xq, &p, 1, inverse);
        let y2 = run_c32(&Fallback, &x, &p, 2, inverse);
        let y2q = run_cq15(&Fallback, &xq, &p, 2, inverse);
        for i in 0..x.len() {
            // bits 1: the conjugate of ±1 is itself.
            let neg = p[i] & 1 == 1;
            let want = if neg {
                C32::new(-x[i].re, -x[i].im)
            } else {
                x[i]
            };
            assert_eq!(bits_of(y1[i]), bits_of(want));
            let want = if neg { cq15_chip(xq[i], 1) } else { xq[i] };
            assert_eq!(y1q[i], want);
            // bits 2: j^k, or j^-k for the descrambler.
            let k = u32::from(p[i] & 3);
            let q = if inverse { (4 - k) & 3 } else { k };
            let (re, im) = quadrant_table(x[i].re, x[i].im, q, f32_neg);
            assert_eq!(bits_of(y2[i]), (re.to_bits(), im.to_bits()));
            let (re, im) = quadrant_table(xq[i].re, xq[i].im, q, q15_neg_sat);
            assert_eq!(y2q[i], CQ15 { re, im });
        }
    }
}

#[test]
fn the_round_trip_at_bits_1_and_2_is_exact_except_minus_one() {
    // Every Q15 value in each component, every index.
    for bits in [1u32, 2] {
        for k in 0..(1u8 << bits) {
            let (q, _) = split_index(u32::from(k), bits);
            let re_neg = matches!(q & 3, 2 | 3);
            let im_neg = matches!(q & 3, 1 | 2);
            for w in [0i16, -1, 777, i16::MIN, i16::MAX] {
                let x: Vec<CQ15> = (i16::MIN..=i16::MAX)
                    .flat_map(|v| [cq(v, w), cq(w, v)])
                    .collect();
                let p = vec![k; x.len()];
                let y = run_cq15(&Fallback, &x, &p, bits, false);
                let back = run_cq15(&Fallback, &y, &p, bits, true);
                for (a, b) in x.iter().zip(&back) {
                    let want_re = if re_neg && a.re.0 == i16::MIN {
                        -32767
                    } else {
                        a.re.0
                    };
                    let want_im = if im_neg && a.im.0 == i16::MIN {
                        -32767
                    } else {
                        a.im.0
                    };
                    assert_eq!(*b, cq(want_re, want_im), "bits {bits}, k {k}, x {a:?}");
                }
            }
        }
    }
    // bits 1, k = 1 changes -32768 in both components.
    let y = run_cq15(&Fallback, &[cq(-32768, -32768)], &[1], 1, false);
    assert_eq!(run_cq15(&Fallback, &y, &[1], 1, true), [cq(-32767, -32767)]);
    // C32: exact by bits for every special value and random values.
    let mut x: Vec<C32> = Vec::new();
    for re in specials() {
        for im in specials() {
            x.push(C32::new(re, im));
        }
    }
    let mut rng = Rng::new(SEED + 5);
    x.extend((0..1000).map(|_| rng.c32()));
    let p: Vec<u8> = (0..x.len()).map(|_| rng.byte()).collect();
    for bits in [1, 2] {
        let y = run_c32(&Fallback, &x, &p, bits, false);
        let back = run_c32(&Fallback, &y, &p, bits, true);
        for (a, b) in x.iter().zip(&back) {
            assert_eq!(bits_of(*a), bits_of(*b));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The direct lookup and the relation to `rotate`.

#[test]
fn no_entry_of_the_f32_cosine_table_is_zero() {
    assert!(COS_F32.iter().all(|v| v.to_bits() & 0x7fff_ffff != 0));
}

#[test]
fn the_direct_lookup_equals_the_rotate_interpolation_for_every_residue() {
    for bits in 3..=8u32 {
        for r in 1..(1u32 << (bits - 2)) {
            let i = (r << (12 - bits)) as usize;
            let phase = r << (32 - bits);
            assert_eq!(phase & 0xF_FFFF, 0);
            let y = rotate_c32(C32::new(1.0, 0.0), phase);
            assert_eq!(
                bits_of(y),
                (COS_F32[i].to_bits(), COS_F32[i + 3072].to_bits()),
                "bits {bits}, r {r}"
            );
            assert_eq!(residual_c32(r, bits), Some((COS_F32[i], COS_F32[i + 3072])));
            let y = rotate_cq15(cq(-32768, 0), phase);
            assert_eq!(y, cq(-COS_Q15[i], -COS_Q15[i + 3072]), "bits {bits}, r {r}");
            assert_eq!(residual_q15(r, bits), Some((COS_Q15[i], COS_Q15[i + 3072])));
            assert!(COS_Q15[i] > 0 && COS_Q15[i + 3072] > 0);
            assert!(COS_F32[i] > 0.0 && COS_F32[i + 3072] > 0.0);
        }
        assert_eq!(residual_c32(0, bits), None);
        assert_eq!(residual_c32(1 << (bits - 2), bits), None);
    }
    assert_eq!(residual_c32(1, 2), None);
    assert_eq!(residual_q15(1, 9), None);
}

#[test]
fn a_residue_is_one_rotate_step_followed_by_the_quadrant_map() {
    let mut rng = Rng::new(SEED + 6);
    for bits in [3u32, 8] {
        let n = 1usize << bits;
        for _ in 0..40 {
            let x: Vec<C32> = (0..n).map(|_| rng.c32()).collect();
            let xq: Vec<CQ15> = (0..n).map(|_| rng.cq15()).collect();
            let p: Vec<u8> = (0..n).map(|k| k as u8).collect();
            let y = run_c32(&Fallback, &x, &p, bits, false);
            let yq = run_cq15(&Fallback, &xq, &p, bits, false);
            for k in 0..n as u32 {
                let (q, r) = split_index(k, bits);
                let (want, wantq) = if r == 0 {
                    (
                        c32_quadrant(x[k as usize], q),
                        cq15_quadrant(xq[k as usize], q),
                    )
                } else {
                    let phase = r << (32 - bits);
                    (
                        c32_quadrant(rotate_c32(x[k as usize], phase), q),
                        cq15_quadrant(rotate_cq15(xq[k as usize], phase), q),
                    )
                };
                assert_eq!(bits_of(y[k as usize]), bits_of(want), "bits {bits}, k {k}");
                assert_eq!(yq[k as usize], wantq, "bits {bits}, k {k}");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The inverse index and the round trips at bits 3 to 8.

#[test]
fn the_inverse_index_is_minus_k_mod_2_pow_bits() {
    let mut rng = Rng::new(SEED + 7);
    let x: Vec<C32> = (0..2000).map(|_| rng.c32()).collect();
    let xq: Vec<CQ15> = (0..2000).map(|_| rng.cq15()).collect();
    let p: Vec<u8> = (0..2000).map(|_| rng.byte()).collect();
    for bits in 1..=8u32 {
        let neg: Vec<u8> = p
            .iter()
            .map(|&v| (0u8.wrapping_sub(v)) & ((1u16 << bits) - 1) as u8)
            .collect();
        for (a, b) in run_c32(&Fallback, &x, &p, bits, true)
            .iter()
            .zip(&run_c32(&Fallback, &x, &neg, bits, false))
        {
            assert_eq!(bits_of(*a), bits_of(*b));
        }
        assert_eq!(
            run_cq15(&Fallback, &xq, &p, bits, true),
            run_cq15(&Fallback, &xq, &neg, bits, false)
        );
    }
}

#[test]
fn the_round_trip_at_bits_3_to_8_stays_within_2_lsb_and_120_db() {
    const N: usize = 200_000;
    let mut rng = Rng::new(SEED + 8);
    let mut worst = 0i32;
    for bits in 3..=8u32 {
        let xq: Vec<CQ15> = (0..N).map(|_| rng.cq15_disk()).collect();
        let p: Vec<u8> = (0..N).map(|_| rng.byte()).collect();
        let y = run_cq15(&Fallback, &xq, &p, bits, false);
        let back = run_cq15(&Fallback, &y, &p, bits, true);
        let mut max = 0i32;
        for (a, b) in xq.iter().zip(&back) {
            let e = (i32::from(a.re.0) - i32::from(b.re.0))
                .abs()
                .max((i32::from(a.im.0) - i32::from(b.im.0)).abs());
            max = max.max(e);
        }
        eprintln!(
            "scramble CQ15 round trip, bits {bits}: max error {max} LSB, SNR {:.1} dB",
            snr_db_q15(&xq, &back)
        );
        assert!(max <= 2, "bits {bits}: {max} LSB");
        worst = worst.max(max);

        let x: Vec<C32> = (0..N).map(|_| rng.c32()).collect();
        let y = run_c32(&Fallback, &x, &p, bits, false);
        let back = run_c32(&Fallback, &y, &p, bits, true);
        let snr = snr_db(&x, &back);
        eprintln!("scramble C32 round trip, bits {bits}: SNR {snr:.1} dB");
        assert!(snr >= 120.0, "bits {bits}: {snr} dB");
    }
    eprintln!("scramble CQ15 round trip, bits 3 to 8: max error {worst} LSB");
}

// ---------------------------------------------------------------------------------------------
// Fallback against Reference, the per-sample helpers.

/// `Fallback` against `Reference`: bit for bit where `r = 0` (a `C32` NaN by class), within the
/// contract SNR elsewhere (finite values).
fn fallback_vs_reference(x: &[C32], xq: &[CQ15], p: &[u8]) {
    for bits in 1..=8u32 {
        for inverse in [false, true] {
            let f = run_c32(&Fallback, x, p, bits, inverse);
            let r = run_c32(&Reference, x, p, bits, inverse);
            let fq = run_cq15(&Fallback, xq, p, bits, inverse);
            let rq = run_cq15(&Reference, xq, p, bits, inverse);
            let (mut wf, mut wr, mut wfq, mut wrq) = (vec![], vec![], vec![], vec![]);
            for i in 0..x.len() {
                let k = phase_index(p[i], bits, inverse);
                let (_, res) = split_index(k, bits);
                if res == 0 {
                    for (a, b) in [(f[i].re, r[i].re), (f[i].im, r[i].im)] {
                        if a.is_nan() || b.is_nan() {
                            assert!(a.is_nan() && b.is_nan(), "bits {bits}, sample {i}");
                        } else {
                            assert_eq!(a.to_bits(), b.to_bits(), "bits {bits}, sample {i}");
                        }
                    }
                    assert_eq!(fq[i], rq[i], "bits {bits}, sample {i}");
                } else {
                    if x[i].re.is_finite() && x[i].im.is_finite() {
                        wf.push(f[i]);
                        wr.push(r[i]);
                    }
                    wfq.push(fq[i]);
                    wrq.push(rq[i]);
                }
                // The per-sample helpers equal the kernel.
                assert_eq!(
                    bits_of(scramble_one_c32(x[i], k, bits).unwrap()),
                    bits_of(f[i])
                );
                assert_eq!(scramble_one_cq15(xq[i], k, bits).unwrap(), fq[i]);
            }
            if !wr.is_empty() && wr.iter().any(|v| v.re != 0.0 || v.im != 0.0) {
                let snr = snr_db(&wr, &wf);
                assert!(snr >= 120.0, "C32 bits {bits}: {snr} dB");
            }
            if !wrq.is_empty() && wrq.iter().any(|v| v.re.0 != 0 || v.im.0 != 0) {
                let snr = snr_db_q15(&wrq, &wfq);
                assert!(snr >= 80.0, "CQ15 bits {bits}: {snr} dB");
            }
        }
    }
}

#[test]
fn fallback_matches_the_reference_on_random_and_special_inputs() {
    let mut rng = Rng::new(SEED + 9);
    let n = 4096;
    let p: Vec<u8> = (0..n).map(|_| rng.byte()).collect();
    // Random input.
    let x: Vec<C32> = (0..n).map(|_| rng.c32()).collect();
    let xq: Vec<CQ15> = (0..n).map(|_| rng.cq15()).collect();
    fallback_vs_reference(&x, &xq, &p);
    // Silence.
    fallback_vs_reference(&vec![C32::new(0.0, 0.0); n], &vec![cq(0, 0); n], &p);
    // Full scale.
    let x: Vec<C32> = (0..n)
        .map(|i| {
            C32::new(
                if i % 2 == 0 { 1.0 } else { -1.0 },
                if i % 3 == 0 { -1.0 } else { 1.0 },
            )
        })
        .collect();
    let corners = [
        cq(-32768, -32768),
        cq(-32768, 32767),
        cq(32767, -32768),
        cq(32767, 32767),
    ];
    let xq: Vec<CQ15> = (0..n).map(|i| corners[i % 4]).collect();
    fallback_vs_reference(&x, &xq, &p);
    // NaN and infinities (C32), mixed with ordinary values.
    let sp: Vec<f32> = specials().collect();
    let x: Vec<C32> = (0..n)
        .map(|i| {
            if i % 3 == 0 {
                C32::new(sp[i % sp.len()], sp[(i / 7) % sp.len()])
            } else {
                rng.c32()
            }
        })
        .collect();
    let xq: Vec<CQ15> = (0..n).map(|_| rng.cq15()).collect();
    fallback_vs_reference(&x, &xq, &p);
}

// ---------------------------------------------------------------------------------------------
// Block-size invariance (test 3).

#[test]
fn the_output_does_not_depend_on_the_split_into_calls() {
    let mut rng = Rng::new(PLAN_SEED);
    let n = 3000;
    let x: Vec<C32> = (0..n).map(|_| rng.c32()).collect();
    let xq: Vec<CQ15> = (0..n).map(|_| rng.cq15()).collect();
    let p: Vec<u8> = (0..n).map(|_| rng.byte()).collect();
    for bits in [1u32, 2, 3, 5, 8] {
        for inverse in [false, true] {
            let whole = run_c32(&Fallback, &x, &p, bits, inverse);
            let wholeq = run_cq15(&Fallback, &xq, &p, bits, inverse);
            for plan in 0..20u64 {
                let mut prng = Rng::new(PLAN_SEED * 100 + plan);
                let mut y = vec![C32::default(); n];
                let mut yq = vec![CQ15::default(); n];
                let mut pos = 0;
                while pos < n {
                    let len = (1 + prng.below(300) as usize).min(n - pos);
                    let e = pos + len;
                    Fallback
                        .scramble(&x[pos..e], &p[pos..e], bits, inverse, &mut y[pos..e])
                        .unwrap();
                    Fallback
                        .scramble(&xq[pos..e], &p[pos..e], bits, inverse, &mut yq[pos..e])
                        .unwrap();
                    pos = e;
                }
                assert!(
                    whole
                        .iter()
                        .zip(&y)
                        .all(|(a, b)| bits_of(*a) == bits_of(*b))
                );
                assert_eq!(wholeq, yq, "bits {bits}, plan {plan}");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Extremes (test 4).

#[test]
fn an_empty_input_is_accepted() {
    for bits in 1..=8 {
        Fallback
            .scramble(&[] as &[C32], &[], bits, false, &mut [])
            .unwrap();
        Reference
            .scramble(&[] as &[CQ15], &[], bits, true, &mut [])
            .unwrap();
    }
}

#[test]
fn every_index_at_bits_8_in_both_directions() {
    let mut rng = Rng::new(SEED + 10);
    let p: Vec<u8> = (0..=255).collect();
    for _ in 0..16 {
        let x: Vec<C32> = (0..256).map(|_| rng.c32()).collect();
        let xq: Vec<CQ15> = (0..256).map(|_| rng.cq15_disk()).collect();
        fallback_vs_reference(&x, &xq, &p);
        for inverse in [false, true] {
            let y = run_c32(&Fallback, &x, &p, 8, inverse);
            let yq = run_cq15(&Fallback, &xq, &p, 8, inverse);
            for k in 0..256usize {
                // Magnitude kept: |y| = |x| up to the table error.
                let (a, b) = (x[k].norm_sqr().sqrt(), y[k].norm_sqr().sqrt());
                assert!((a - b).abs() <= 1e-6, "k {k}: {a} {b}");
                let kk = phase_index(k as u8, 8, inverse);
                assert_eq!(scramble_one_cq15(xq[k], kk, 8).unwrap(), yq[k]);
            }
        }
    }
}

#[test]
fn full_scale_cq15_saturates_like_rotate_and_negates_saturating() {
    let corners = [
        cq(-32768, -32768),
        cq(-32768, 32767),
        cq(32767, -32768),
        cq(32767, 32767),
    ];
    for bits in 1..=8u32 {
        for k in 0..(1u32 << bits) {
            let (q, r) = split_index(k, bits);
            for &x in &corners {
                let y = run_cq15(&Fallback, &[x], &[k as u8], bits, false)[0];
                let want = if r == 0 {
                    cq15_quadrant(x, q)
                } else {
                    cq15_quadrant(rotate_cq15(x, r << (32 - bits)), q)
                };
                assert_eq!(y, want, "bits {bits}, k {k}, x {x:?}");
            }
        }
    }
    // The saturating negation: -1 · (-1, -1) = (32767, 32767), j · (-1, -1) = (32767, -32768).
    assert_eq!(
        run_cq15(&Fallback, &[cq(-32768, -32768)], &[2], 2, false),
        [cq(32767, 32767)]
    );
    assert_eq!(
        run_cq15(&Fallback, &[cq(-32768, -32768)], &[1], 2, false),
        [cq(32767, -32768)]
    );
    // A residual product of a full-scale sample saturates: (1, 1) · e^{jπ/8} at bits 4.
    let y = run_cq15(&Fallback, &[cq(32767, 32767)], &[1], 4, false)[0];
    assert_eq!(y.im, Q15(32767));
}

#[test]
fn infinities_stay_infinite_where_the_residue_is_0() {
    let inf = f32::INFINITY;
    let x = [
        C32::new(inf, 1.0),
        C32::new(-inf, -0.0),
        C32::new(0.0, inf),
        C32::new(inf, -inf),
        C32::new(f32::NAN, 2.0),
    ];
    for bits in 1..=8u32 {
        for k in 0..(1u32 << bits) {
            let (q, r) = split_index(k, bits);
            let p = vec![k as u8; x.len()];
            let y = run_c32(&Fallback, &x, &p, bits, false);
            for (a, b) in x.iter().zip(&y) {
                if r == 0 {
                    let (wr, wi) = quadrant_table(a.re, a.im, q, f32_neg);
                    assert_eq!(bits_of(*b), (wr.to_bits(), wi.to_bits()));
                    let infinite =
                        |v: C32| u32::from(v.re.is_infinite()) + u32::from(v.im.is_infinite());
                    assert_eq!(infinite(*a), infinite(*b));
                } else if a.re.is_infinite() && a.im.is_finite() {
                    // One infinite component gives infinite components (c, s > 0).
                    assert!(
                        b.re.is_infinite() && b.im.is_infinite(),
                        "bits {bits}, k {k}"
                    );
                } else if a.re.is_nan() {
                    assert!(b.re.is_nan() && b.im.is_nan());
                }
            }
        }
    }
}

#[test]
fn the_kernel_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Fallback>();
    assert_send_sync::<Reference>();
}
