//! Tests of the M48 kernel `gauss_noise` of `sdrk-core` (tests 2 and 6 of the milestone): the
//! construction rules of `GaussNoiseState`, `set_sigma`, the index after calls, `seek` against
//! generating and discarding (indices 0, 1, 4095 and 2^32 + 3, draws per sample 1 and 2),
//! `reset`, an empty output, the `InvalidState` of a type change, the PCG32 known answer and the
//! interleaved chains, the properties of the inverse-CDF table and of `normal_q17` (continuity,
//! monotonicity, symmetry), the output formats against the exact `f64` value, chunk invariance
//! (seed 687), the extremes (`sigma` 0, 1e-30, 9e-31, 16, the index `u64::MAX`), timing and the
//! thread-safety of the public types. Seeds 680 to 689 belong to M48; these tests use the
//! generator seed 680 and the chunk-plan seed 687. The comparison of `Fallback` with `Reference`
//! for every type runs only with the feature `reference`.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::useless_vec
)]

use sdrk_core::kernels::noise::{
    GaussNoise, GaussNoiseState, PCG32_MULT, draws_per_sample, gauss_icdf_table, normal_q17,
    pcg32_output,
};
use sdrk_core::num::q15_from_f64;
use sdrk_core::{C32, CQ15, Delay, Error, Fallback, Q15, Ratio, SampleType, Timing};
use sdrk_device::{ErrorKind, SdrkError};

/// The generator seed of these tests (M48 range 680 to 689).
const SEED: u64 = 680;
/// The stream of these tests.
const STREAM: u64 = 1;
/// The seed of the chunk plans; plan `k` uses `SEED_PLANS * 100 + k`.
const SEED_PLANS: u64 = 687;
/// Chunk plans per type.
const PLANS: u64 = 20;
/// `2^-17`.
const Q17: f64 = 1.0 / 131_072.0;

// ------------------------------------------------------------------ independent model

/// PCG32 of `pcg_basic.c`, written here as an affine map of the LCG so that the jump is computed
/// by composing maps instead of the `advance` loop of the kernel.
#[derive(Clone, Copy)]
struct Model {
    state: u64,
    inc: u64,
}

impl Model {
    fn new(seed: u64, stream: u64) -> Self {
        let inc = (stream << 1) | 1;
        let mut state = inc;
        state = state.wrapping_add(seed);
        state = state.wrapping_mul(PCG32_MULT).wrapping_add(inc);
        Self { state, inc }
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(PCG32_MULT).wrapping_add(self.inc);
        let x = (((old >> 18) ^ old) >> 27) as u32;
        x.rotate_right((old >> 59) as u32)
    }

    /// Skips `n` draws: the map `x -> a x + c` raised to the power `n` by squaring.
    fn jump(&mut self, mut n: u64) {
        let (mut ra, mut rc) = (1u64, 0u64);
        let (mut a, mut c) = (PCG32_MULT, self.inc);
        while n != 0 {
            if n & 1 == 1 {
                rc = a.wrapping_mul(rc).wrapping_add(c);
                ra = a.wrapping_mul(ra);
            }
            c = a.wrapping_mul(c).wrapping_add(c);
            a = a.wrapping_mul(a);
            n >>= 1;
        }
        self.state = ra.wrapping_mul(self.state).wrapping_add(rc);
    }

    /// The model positioned at sample `n` of a type with `d` draws per sample.
    fn at(seed: u64, stream: u64, n: u64, d: u64) -> Self {
        let mut m = Self::new(seed, stream);
        m.jump(n.wrapping_mul(d));
        m
    }
}

/// The exact value `v = zq · 2^-17 · sigma_e` of one draw (`sigma_e` = `sigma`, 0 below 1e-30).
fn exact(r: u32, sigma: f32) -> f64 {
    let s = if sigma < 1e-30 { 0.0 } else { f64::from(sigma) };
    f64::from(normal_q17(r)) * Q17 * s
}

/// The `f32` output of the contract: `v` rounded once, `+0.0` for a flushed sigma.
fn f32_of(r: u32, sigma: f32) -> f32 {
    let v = exact(r, sigma);
    if v == 0.0 { 0.0 } else { v as f32 }
}

/// The `Q15` output of the contract: R7 applied to `v`.
fn q15_of(r: u32, sigma: f32) -> Q15 {
    q15_from_f64(exact(r, sigma))
}

fn bits_c(v: &[C32]) -> Vec<(u32, u32)> {
    v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

fn bits_r(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn state(sigma: f32) -> GaussNoiseState {
    GaussNoiseState::new(SEED, STREAM, sigma).unwrap()
}

/// Deterministic chunk sizes (xorshift64*) in `1..=max` summing to `n`.
fn plan(seed: u64, n: usize, max: usize) -> Vec<usize> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut left = n;
    let mut sizes = Vec::new();
    while left > 0 {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        let r = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize;
        let k = (1 + r % max).min(left);
        sizes.push(k);
        left -= k;
    }
    sizes
}

/// Base and slope of table entry `i`.
fn entry(i: usize) -> (u32, u32) {
    let e = gauss_icdf_table()[i];
    (e >> 12, e & 0xFFF)
}

/// The draw `r` (sign bit clear) whose `x = 2r + 1` lies in octave `l`, segment `sub`, at
/// `frac`, with the bits below `frac` set to `low`; `None` when the octave has too few bits
/// for that position.
fn draw_at(l: u32, sub: u32, frac: u32, low: u32) -> Option<u32> {
    // n = 1 . sub(5) . frac(15) . rest(11) at bits 31 ..= 0; x = n >> (31 - l), made odd.
    let n = (1u64 << 31) | u64::from(sub) << 26 | u64::from(frac) << 11 | u64::from(low & 0x7FF);
    let x = (n >> (31 - l)) as u32 | 1;
    let back = x << x.leading_zeros();
    ((back >> 26) & 31 == sub && (back >> 11) & 0x7FFF == frac).then_some(x >> 1)
}

// ------------------------------------------------------------------ construction

#[test]
fn construction_checks_sigma() {
    for bad in [
        -0.1f32,
        -f32::MIN_POSITIVE,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        16.000_002,
        17.0,
    ] {
        let e = GaussNoiseState::new(1, 1, bad).unwrap_err();
        assert!(matches!(e, Error::InvalidParameter(_)), "{bad}");
        assert_eq!(e.kind(), ErrorKind::InvalidArgument);
    }
    for good in [0.0f32, 9e-31, 1e-30, 0.1, 1.0, 16.0] {
        let s = GaussNoiseState::new(1, 1, good).unwrap();
        assert_eq!(s.sigma(), good, "the getter returns the value that was set");
        assert_eq!(s.index(), 0);
        assert_eq!(s.sample_type(), None);
    }
    // -0.0 counts as +0.0.
    let s = GaussNoiseState::new(1, 1, -0.0).unwrap();
    assert_eq!(s.sigma().to_bits(), 0);
    let s = GaussNoiseState::new(3, 4, 0.5).unwrap();
    assert_eq!((s.seed(), s.stream()), (3, 4));
}

#[test]
fn the_scale_follows_sigma() {
    let s = state(1.0).scale();
    assert_eq!(s.s17(), 1.0 / 131_072.0);
    assert_eq!((s.ms(), s.sh()), (0x80_0000, 25));
    let s = state(16.0).scale();
    assert_eq!((s.ms(), s.sh()), (0x80_0000, 21));
    // The f32 just below 16: the largest significand at the largest exponent below.
    let s = state(f32::from_bits(16.0f32.to_bits() - 1)).scale();
    assert_eq!((s.ms(), s.sh()), (0xFF_FFFF, 22));
    // sh ≥ 45 gives ms = 0 (every Q15 output is 0), but the f32 scale stays.
    let s = state(1e-12).scale();
    assert_eq!((s.ms(), s.sh()), (0, 21));
    assert_eq!(s.s17(), 1e-12 / 131_072.0);
    assert!(!s.is_zero());
    // Flushed and zero sigma.
    for z in [0.0, 9e-31] {
        let s = state(z).scale();
        assert!(s.is_zero());
        assert_eq!((s.s17().to_bits(), s.ms(), s.sh()), (0, 0, 21));
    }
    assert!(!state(1e-30).scale().is_zero());
}

#[test]
fn set_sigma_checks_recomputes_the_scale_and_keeps_the_position() {
    let mut s = state(0.5);
    let mut first = [0.0f32; 10];
    Fallback.gauss_noise(&mut s, &mut first).unwrap();
    for bad in [-1.0f32, f32::NAN, f32::INFINITY, 16.5] {
        assert!(matches!(s.set_sigma(bad), Err(Error::InvalidParameter(_))));
        assert_eq!(s.sigma(), 0.5, "an error changes nothing");
        assert_eq!(s.scale(), state(0.5).scale());
    }
    s.set_sigma(0.25).unwrap();
    assert_eq!(s.sigma(), 0.25);
    assert_eq!(s.scale(), state(0.25).scale());
    assert_eq!(s.index(), 10);
    let mut next = [0.0f32; 6];
    Fallback.gauss_noise(&mut s, &mut next).unwrap();
    // The same sequence position, the new level.
    let mut m = Model::at(SEED, STREAM, 10, 1);
    for (k, &x) in next.iter().enumerate() {
        assert_eq!(
            x.to_bits(),
            f32_of(m.next_u32(), 0.25).to_bits(),
            "sample {}",
            10 + k
        );
    }
}

#[test]
fn the_index_counts_the_samples_of_every_call() {
    let mut s = state(1.0);
    let mut buf = [C32::default(); 100];
    for (k, n) in [1usize, 7, 0, 100, 33].into_iter().enumerate() {
        let before = s.index();
        Fallback.gauss_noise(&mut s, &mut buf[..n]).unwrap();
        assert_eq!(s.index(), before + n as u64, "call {k}");
    }
    assert_eq!(s.index(), 141);
}

// ------------------------------------------------------------------ generator

#[test]
fn pcg32_gives_the_known_answer_of_pcg_basic() {
    let mut s = GaussNoiseState::new(42, 54, 1.0).unwrap();
    let (rng, _) = s.begin(SampleType::F32, 6).unwrap().unwrap();
    let want = [
        0xa15c_02b7_u32,
        0x7b47_f409,
        0xba1d_3330,
        0x83d2_f293,
        0xbfa4_784b,
        0xcbed_606e,
    ];
    for w in want {
        assert_eq!(rng.next_u32(), w);
    }
}

#[test]
fn draws_per_sample_is_one_per_real_value() {
    assert_eq!(draws_per_sample(SampleType::C32), 2);
    assert_eq!(draws_per_sample(SampleType::CQ15), 2);
    assert_eq!(draws_per_sample(SampleType::F32), 1);
    assert_eq!(draws_per_sample(SampleType::Q15), 1);
}

#[test]
fn interleaved_chains_equal_the_single_chain() {
    for len in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 63, 64, 1001] {
        let mut a = state(1.0);
        let (ra, _) = a.begin(SampleType::F32, 1).unwrap().unwrap();
        let start = *ra;
        let single: Vec<u32> = (0..len).map(|_| ra.next_u32()).collect();
        let end = *ra;
        let mut r2 = start;
        let mut x2 = vec![0u32; len];
        r2.fill_interleaved::<2>(&mut x2);
        assert_eq!((&x2, r2), (&single, end), "k 2, len {len}");
        let mut r4 = start;
        let mut x4 = vec![0u32; len];
        r4.fill_interleaved::<4>(&mut x4);
        assert_eq!((&x4, r4), (&single, end), "k 4, len {len}");
        let mut r1 = start;
        let mut x1 = vec![0u32; len];
        r1.fill_interleaved::<1>(&mut x1);
        assert_eq!((&x1, r1), (&single, end), "k 1, len {len}");
    }
}

#[test]
fn jump_and_output_agree_with_stepping() {
    let mut s = state(1.0);
    let (rng, _) = s.begin(SampleType::F32, 1).unwrap().unwrap();
    for k in [0u64, 1, 2, 4, 8, 16, 1000] {
        let (a, c) = rng.jump(k);
        let mut b = *rng;
        for _ in 0..k {
            b.next_u32();
        }
        assert_eq!(
            a.wrapping_mul(rng.state()).wrapping_add(c),
            b.state(),
            "k {k}"
        );
        assert_eq!(b.increment(), (STREAM << 1) | 1);
        let mut c2 = *rng;
        c2.advance(k);
        assert_eq!(c2, b);
    }
    let at = rng.state();
    assert_eq!(pcg32_output(at), rng.next_u32());
}

// ------------------------------------------------------------------ table and inverse CDF

#[test]
fn the_table_is_continuous_monotonic_and_bounded() {
    let t = gauss_icdf_table();
    assert_eq!(t.len(), 1024);
    assert_eq!(entry(0).0, 830_729);
    for i in 0..1024 {
        let (base, slope) = entry(i);
        assert!((1..=4095).contains(&slope), "slope {slope} at {i}");
        assert!(base < 1 << 20, "base at {i}");
        let next = if i == 1023 { 0 } else { entry(i + 1).0 };
        // Continuity within and across octaves, and the last knot is 0.
        assert_eq!(base - slope, next, "continuity at {i}");
        assert!(base >= next, "non-increasing at {i}");
    }
    // The measured slope range of the contract.
    let slopes: Vec<u32> = (0..1024).map(|i| entry(i).1).collect();
    assert_eq!(*slopes.iter().min().unwrap(), 323);
    assert_eq!(*slopes.iter().max().unwrap(), 3197);
}

#[test]
fn normal_q17_is_non_increasing_in_x() {
    let mut last = i32::MAX;
    let mut tested = 0;
    for l in 0..32 {
        for sub in 0..32 {
            for frac in [0u32, 1, 16_384, 32_766, 32_767] {
                let (Some(lo), Some(hi)) = (draw_at(l, sub, frac, 0), draw_at(l, sub, frac, 0x7FF))
                else {
                    continue;
                };
                // The bits below frac do not matter.
                let z = normal_q17(lo);
                assert_eq!(z, normal_q17(hi), "L {l} sub {sub} frac {frac}");
                assert!(z <= last, "L {l} sub {sub} frac {frac}: {z} > {last}");
                assert!(z >= 0);
                if frac == 0 {
                    // frac 0 is the knot itself.
                    assert_eq!(z as u32, entry(((l << 5) | sub) as usize).0);
                }
                last = z;
                tested += 1;
            }
        }
    }
    // Octaves 20 to 31 have 20 or more bits below the leading one, so every segment holds all
    // five positions; below, `x` is too short (and odd), and the dense sweep covers them.
    assert!(tested >= 12 * 32 * 5, "{tested}");
    // The extremes of the magnitude.
    assert_eq!(normal_q17(0), 830_729);
    assert_eq!(normal_q17(0x7FFF_FFFF), 0);
    // Every draw of the octaves 0 to 20 (r below 2^20), ascending in x, is non-increasing.
    let mut last = i32::MAX;
    for r in 0..1u32 << 20 {
        let z = normal_q17(r);
        assert!(z <= last, "r {r:#x}");
        last = z;
    }
    // A dense sweep over all octaves, ascending in x, is non-increasing.
    let mut last = i32::MAX;
    let mut r = 0u32;
    while r < 0x8000_0000 {
        let z = normal_q17(r);
        assert!(z <= last, "r {r:#x}");
        last = z;
        r += 0x1_0001;
    }
}

#[test]
fn normal_q17_is_symmetric() {
    let mut m = Model::new(SEED, 9);
    for _ in 0..100_000 {
        let r = m.next_u32();
        assert_eq!(normal_q17(r ^ 0x8000_0000), -normal_q17(r), "r {r:#x}");
    }
    for r in [0u32, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF] {
        assert_eq!(normal_q17(r ^ 0x8000_0000), -normal_q17(r), "r {r:#x}");
    }
    assert_eq!(normal_q17(0x8000_0000), -830_729);
    assert_eq!(normal_q17(0xFFFF_FFFF), 0);
}

#[test]
fn normal_q17_is_close_to_the_inverse_cdf() {
    // -Φ⁻¹((2r + 1) / 2^33) from `scipy.special.ndtri`.
    let known = [
        (1u32 << 30, 0.674_489_749_829_738_4),
        (1 << 26, 2.153_874_691_093_316),
        (1 << 21, 3.297_193_278_732_375),
        (1 << 11, 4.900_916_256_001_619_5),
        (2, 6.085_104_545_783_031),
        (0, 6.337_957_754_553_789_5),
        (0x7FFF_FFFE, 8.754_298_118_749_869e-10),
    ];
    for (r, z) in known {
        let got = f64::from(normal_q17(r)) * Q17;
        assert!((got - z).abs() < 5.7e-5, "r {r:#x}: {got} vs {z}");
    }
}

// ------------------------------------------------------------------ output formats

#[test]
fn the_output_formats_round_the_exact_value_once() {
    const N: usize = 100_000;
    let sigmas = [
        f32::from_bits(0x3580_0000), // 2^-20
        1e-4,
        0.25,
        0.6,
        1.0,
        4.0,
        f32::from_bits(16.0f32.to_bits() - 1),
        16.0,
    ];
    for (k, &sigma) in sigmas.iter().enumerate() {
        let seed = SEED + k as u64;
        let mut r = vec![0.0f32; N];
        Fallback
            .gauss_noise(&mut GaussNoiseState::new(seed, 5, sigma).unwrap(), &mut r)
            .unwrap();
        let mut q = vec![Q15(0); N];
        Fallback
            .gauss_noise(&mut GaussNoiseState::new(seed, 5, sigma).unwrap(), &mut q)
            .unwrap();
        let mut m = Model::new(seed, 5);
        for n in 0..N {
            let d = m.next_u32();
            let v = exact(d, sigma);
            assert_eq!(
                r[n].to_bits(),
                (v as f32).to_bits(),
                "f32 sigma {sigma} n {n}"
            );
            assert_eq!(q[n], q15_from_f64(v), "q15 sigma {sigma} n {n}");
        }
    }
    // sh ≥ 45: sigma 1e-12 gives all Q15 zeros, and nonzero f32 values.
    let mut q = vec![Q15(1); 4096];
    Fallback.gauss_noise(&mut state(1e-12), &mut q).unwrap();
    assert!(q.iter().all(|x| x.0 == 0));
    let mut cq = vec![CQ15::default(); 4096];
    Fallback.gauss_noise(&mut state(1e-12), &mut cq).unwrap();
    assert!(cq.iter().all(|x| *x == CQ15::default()));
    let mut r = vec![0.0f32; 4096];
    Fallback.gauss_noise(&mut state(1e-12), &mut r).unwrap();
    assert!(r.iter().filter(|x| **x != 0.0).count() > 4000);
}

#[test]
fn values_match_the_model_for_every_type() {
    const N: usize = 512;
    let sigma = 0.7f32;
    let mut c = vec![C32::default(); N];
    Fallback.gauss_noise(&mut state(sigma), &mut c).unwrap();
    let mut cq = vec![CQ15::default(); N];
    Fallback.gauss_noise(&mut state(sigma), &mut cq).unwrap();
    let mut r = vec![0.0f32; N];
    Fallback.gauss_noise(&mut state(sigma), &mut r).unwrap();
    let mut q = vec![Q15(0); N];
    Fallback.gauss_noise(&mut state(sigma), &mut q).unwrap();
    let mut mc = Model::new(SEED, STREAM);
    let mut mr = Model::new(SEED, STREAM);
    let mut clipped = 0;
    for n in 0..N {
        let (dre, dim) = (mc.next_u32(), mc.next_u32());
        assert_eq!(
            c[n],
            C32::new(f32_of(dre, sigma), f32_of(dim, sigma)),
            "c32 {n}"
        );
        assert_eq!(
            cq[n],
            CQ15 {
                re: q15_of(dre, sigma),
                im: q15_of(dim, sigma)
            },
            "cq15 {n}"
        );
        let d = mr.next_u32();
        assert_eq!(r[n].to_bits(), f32_of(d, sigma).to_bits(), "f32 {n}");
        assert_eq!(q[n], q15_of(d, sigma), "q15 {n}");
        if exact(d, sigma).abs() >= 1.0 {
            clipped += 1;
            assert!(q[n] == Q15::MAX || q[n] == Q15::MIN);
        }
    }
    assert!(clipped > 0, "sigma 0.7 clips some values");
    // The complex buffer holds the real values in draw order.
    assert_eq!(c[0].re.to_bits(), r[0].to_bits());
    assert_eq!(c[0].im.to_bits(), r[1].to_bits());
    assert_eq!(c[1].re.to_bits(), r[2].to_bits());
}

/// The first values of four golden cases, computed by an independent Python model (PCG32 with
/// integers masked to 64 bits, the table from `scipy.special.ndtri`, §2.1 in integers), pin the
/// kernel to an implementation outside this repository's Rust code.
#[test]
fn first_values_match_the_python_model() {
    let mut c = [C32::default(); 2];
    Fallback
        .gauss_noise(&mut GaussNoiseState::new(680, 1, 1.0).unwrap(), &mut c)
        .unwrap();
    assert_eq!(
        c,
        [
            C32::new(-0.764_610_3, -1.310_592_7),
            C32::new(0.329_986_57, -0.787_765_5)
        ]
    );
    let mut r = [0.0f32; 3];
    Fallback
        .gauss_noise(&mut GaussNoiseState::new(682, 1, 1.0).unwrap(), &mut r)
        .unwrap();
    assert_eq!(r, [0.726_402_3, -0.858_146_67, -0.703_849_8]);
    let mut q = [Q15(0); 4];
    Fallback
        .gauss_noise(&mut GaussNoiseState::new(684, 3, 0.6).unwrap(), &mut q)
        .unwrap();
    assert_eq!(q, [Q15(29_906), Q15(-11_261), Q15(15_151), Q15(-6_454)]);
    let mut cq = [CQ15::default(); 2];
    Fallback
        .gauss_noise(&mut GaussNoiseState::new(683, 2, 0.25).unwrap(), &mut cq)
        .unwrap();
    assert_eq!(
        cq,
        [
            CQ15 {
                re: Q15(-6_421),
                im: Q15(-11_923)
            },
            CQ15 {
                re: Q15(-13_083),
                im: Q15(-57)
            }
        ]
    );
}

#[test]
fn the_streams_differ() {
    let mut a = [0.0f32; 16];
    let mut b = [0.0f32; 16];
    Fallback
        .gauss_noise(&mut GaussNoiseState::new(SEED, 1, 1.0).unwrap(), &mut a)
        .unwrap();
    Fallback
        .gauss_noise(&mut GaussNoiseState::new(SEED, 2, 1.0).unwrap(), &mut b)
        .unwrap();
    assert_ne!(bits_r(&a), bits_r(&b));
}

// ------------------------------------------------------------------ seek and reset

#[test]
fn seek_equals_generating_and_discarding() {
    const N: usize = 4100;
    let mut whole_c = vec![C32::default(); N];
    Fallback.gauss_noise(&mut state(1.0), &mut whole_c).unwrap();
    let mut whole_r = vec![0.0f32; N];
    Fallback.gauss_noise(&mut state(1.0), &mut whole_r).unwrap();
    for start in [0u64, 1, 4095] {
        let i = start as usize;
        let mut s = state(1.0);
        Fallback
            .gauss_noise(&mut s, &mut [C32::default(); 3])
            .unwrap();
        s.seek(start);
        assert_eq!(s.index(), start);
        assert_eq!(s.sample_type(), None);
        let mut c = [C32::default(); 5];
        Fallback.gauss_noise(&mut s, &mut c).unwrap();
        assert_eq!(bits_c(&c), bits_c(&whole_c[i..i + 5]), "c32 seek {start}");
        assert_eq!(s.index(), start + 5);

        // seek clears the type: the same state may now produce real samples.
        s.seek(start);
        let mut r = [0.0f32; 5];
        Fallback.gauss_noise(&mut s, &mut r).unwrap();
        assert_eq!(bits_r(&r), bits_r(&whole_r[i..i + 5]), "f32 seek {start}");
    }
    // 2^32 + 3: generating and discarding the last three samples after a jump to 2^32, and the
    // model's own jump, with 2 and 1 draws per sample.
    let far = (1u64 << 32) + 3;
    let mut s = state(1.0);
    s.seek(1 << 32);
    let mut four = [C32::default(); 4];
    Fallback.gauss_noise(&mut s, &mut four).unwrap();
    s.seek(far);
    let mut one = [C32::default(); 1];
    Fallback.gauss_noise(&mut s, &mut one).unwrap();
    assert_eq!(bits_c(&one), bits_c(&four[3..]));
    let mut m = Model::at(SEED, STREAM, far, 2);
    assert_eq!(
        one[0],
        C32::new(f32_of(m.next_u32(), 1.0), f32_of(m.next_u32(), 1.0))
    );
    let mut s = state(1.0);
    s.seek(1 << 32);
    let mut four = [Q15(0); 4];
    Fallback.gauss_noise(&mut s, &mut four).unwrap();
    s.seek(far);
    let mut q = [Q15(0); 1];
    Fallback.gauss_noise(&mut s, &mut q).unwrap();
    assert_eq!(q[0], four[3]);
    assert_eq!(
        q[0],
        q15_of(Model::at(SEED, STREAM, far, 1).next_u32(), 1.0)
    );
}

#[test]
fn reset_restarts_at_sample_zero() {
    let mut s = state(0.3);
    let mut a = [CQ15::default(); 64];
    Fallback.gauss_noise(&mut s, &mut a).unwrap();
    s.reset();
    assert_eq!(s.index(), 0);
    assert_eq!(s.sample_type(), None);
    assert_eq!(s, state(0.3), "reset gives the new state");
    let mut b = [CQ15::default(); 64];
    Fallback.gauss_noise(&mut s, &mut b).unwrap();
    assert_eq!(a, b);
    // sigma survives the reset.
    s.set_sigma(0.1).unwrap();
    s.reset();
    assert_eq!(s.sigma(), 0.1);
}

// ------------------------------------------------------------------ sigma 0, empty, type

/// `sigma` 0 and a flushed `sigma` give `+0.0` (Q15 0) and keep the sequence aligned.
#[test]
fn zero_and_flushed_sigma_give_positive_zeros_and_advance() {
    for z in [0.0f32, -0.0, 9e-31] {
        let mut s = state(z);
        let mut c = [C32::new(9.0, 9.0); 37];
        Fallback.gauss_noise(&mut s, &mut c).unwrap();
        assert!(c.iter().all(|x| x.re.to_bits() == 0 && x.im.to_bits() == 0));
        assert_eq!(s.index(), 37);
        s.set_sigma(1.0).unwrap();
        let mut next = [C32::default(); 3];
        Fallback.gauss_noise(&mut s, &mut next).unwrap();
        let mut whole = [C32::default(); 40];
        Fallback.gauss_noise(&mut state(1.0), &mut whole).unwrap();
        assert_eq!(bits_c(&next), bits_c(&whole[37..]), "aligned to the index");

        let mut s = state(z);
        let mut r = [1.0f32; 50];
        Fallback.gauss_noise(&mut s, &mut r).unwrap();
        assert!(r.iter().all(|x| x.to_bits() == 0));
        let mut q = [Q15(5); 50];
        Fallback.gauss_noise(&mut state(z), &mut q).unwrap();
        assert!(q.iter().all(|x| *x == Q15(0)));
        let mut cq = [CQ15 {
            re: Q15(5),
            im: Q15(5),
        }; 50];
        Fallback.gauss_noise(&mut state(z), &mut cq).unwrap();
        assert!(cq.iter().all(|x| *x == CQ15::default()));
    }
}

#[test]
fn an_empty_output_changes_nothing() {
    let mut s = state(1.0);
    s.seek(17);
    let before = s;
    Fallback.gauss_noise(&mut s, &mut [] as &mut [C32]).unwrap();
    assert_eq!(s, before);
    assert_eq!(s.sample_type(), None, "an empty call does not fix the type");
    Fallback.gauss_noise(&mut s, &mut [] as &mut [f32]).unwrap();
    assert_eq!(s, before);
}

#[test]
fn a_type_change_is_invalid_state() {
    let mut s = state(1.0);
    Fallback
        .gauss_noise(&mut s, &mut [C32::default(); 4])
        .unwrap();
    assert_eq!(s.sample_type(), Some(SampleType::C32));
    let before = s;
    let e = Fallback
        .gauss_noise(&mut s, &mut [CQ15::default(); 4])
        .unwrap_err();
    assert!(matches!(e, Error::InvalidState(_)));
    assert_eq!(e.kind(), ErrorKind::InvalidState);
    assert_eq!(
        e.to_string(),
        "invalid state: the sample type differs from the previous call"
    );
    assert_eq!(s, before, "nothing changes on the error");
    let mut out = [7.0f32; 4];
    assert!(matches!(
        Fallback.gauss_noise(&mut s, &mut out),
        Err(Error::InvalidState(_))
    ));
    assert_eq!(out, [7.0; 4], "nothing is written");
    // Also an empty call of another type is refused.
    assert!(Fallback.gauss_noise(&mut s, &mut [] as &mut [Q15]).is_err());
    // reset clears the type.
    s.reset();
    Fallback.gauss_noise(&mut s, &mut out).unwrap();
    assert_eq!(s.sample_type(), Some(SampleType::F32));
}

// ------------------------------------------------------------------ chunk invariance

fn chunked<T: Copy + Default>(
    whole: &[T],
    sigma: f32,
    mut run: impl FnMut(&mut GaussNoiseState, &mut [T]),
    same: impl Fn(&[T], &[T]) -> bool,
) {
    for k in 0..PLANS {
        let mut s = state(sigma);
        let mut out = vec![T::default(); whole.len()];
        let mut at = 0;
        for n in plan(SEED_PLANS * 100 + k, whole.len(), 300) {
            run(&mut s, &mut out[at..at + n]);
            at += n;
        }
        assert!(same(&out, whole), "plan {k}");
        assert_eq!(s.index(), whole.len() as u64);
    }
}

#[test]
fn the_output_does_not_depend_on_the_chunking() {
    const N: usize = 2000;
    let sigma = 0.4;
    let mut c = vec![C32::default(); N];
    Fallback.gauss_noise(&mut state(sigma), &mut c).unwrap();
    chunked(
        &c,
        sigma,
        |s, o| Fallback.gauss_noise(s, o).unwrap(),
        |a, b| bits_c(a) == bits_c(b),
    );
    let mut cq = vec![CQ15::default(); N];
    Fallback.gauss_noise(&mut state(sigma), &mut cq).unwrap();
    chunked(
        &cq,
        sigma,
        |s, o| Fallback.gauss_noise(s, o).unwrap(),
        |a, b| a == b,
    );
    let mut r = vec![0.0f32; N];
    Fallback.gauss_noise(&mut state(sigma), &mut r).unwrap();
    chunked(
        &r,
        sigma,
        |s, o| Fallback.gauss_noise(s, o).unwrap(),
        |a, b| bits_r(a) == bits_r(b),
    );
    let mut q = vec![Q15(0); N];
    Fallback.gauss_noise(&mut state(sigma), &mut q).unwrap();
    chunked(
        &q,
        sigma,
        |s, o| Fallback.gauss_noise(s, o).unwrap(),
        |a, b| a == b,
    );
}

// ------------------------------------------------------------------ extremes

#[test]
fn sigma_1e30_is_not_flushed() {
    let mut r = [0.0f32; 64];
    Fallback.gauss_noise(&mut state(1e-30), &mut r).unwrap();
    let mut m = Model::new(SEED, STREAM);
    for x in r {
        let want = f32_of(m.next_u32(), 1e-30);
        assert_eq!(x.to_bits(), want.to_bits());
    }
    assert!(r.iter().filter(|x| **x != 0.0).count() > 60);
    // Every nonzero value is a normal f32 (no flush-to-zero mode can change a bit).
    assert!(r.iter().all(|x| *x == 0.0 || x.is_normal()));
    let mut q = [Q15(1); 64];
    Fallback.gauss_noise(&mut state(1e-30), &mut q).unwrap();
    assert!(q.iter().all(|x| x.0 == 0));
}

#[test]
fn sigma_sixteen_saturates_q15_and_stays_finite_in_f32() {
    const N: usize = 4096;
    let mut q = vec![Q15(0); N];
    Fallback.gauss_noise(&mut state(16.0), &mut q).unwrap();
    let at_limit = q
        .iter()
        .filter(|x| **x == Q15::MAX || **x == Q15::MIN)
        .count();
    // P(|z| < 1/16) is about 5 %.
    assert!(at_limit as f64 > 0.9 * N as f64, "{at_limit}");
    let mut cq = vec![CQ15::default(); N];
    Fallback.gauss_noise(&mut state(16.0), &mut cq).unwrap();
    let at_limit = cq
        .iter()
        .filter(|x| x.re == Q15::MAX || x.re == Q15::MIN)
        .count();
    assert!(at_limit as f64 > 0.9 * N as f64, "{at_limit}");
    let mut r = vec![0.0f32; N];
    Fallback.gauss_noise(&mut state(16.0), &mut r).unwrap();
    assert!(r.iter().all(|x| x.is_finite() && x.abs() <= 16.0 * 6.338));
    let mut c = vec![C32::default(); N];
    Fallback.gauss_noise(&mut state(16.0), &mut c).unwrap();
    assert!(c.iter().all(|x| x.re.is_finite() && x.im.is_finite()));
    assert!(c.iter().any(|x| x.re.abs() > 16.0));
    // The largest magnitude, 830729 · 2^-17 · 16, at r = 0 and r = 0x8000_0000.
    let s = state(16.0).scale();
    assert_eq!(
        f64::from(s.f32_value(normal_q17(0))),
        830_729.0 * Q17 * 16.0
    );
    assert_eq!(
        s.f32_value(normal_q17(0x8000_0000)),
        -s.f32_value(normal_q17(0))
    );
    assert_eq!(s.q15_value(normal_q17(0)), Q15::MAX);
    assert_eq!(s.q15_value(normal_q17(0x8000_0000)), Q15::MIN);
}

#[test]
fn the_largest_index_seeks_and_wraps() {
    // Real: sample u64::MAX is draw u64::MAX.
    let mut s = state(1.0);
    s.seek(u64::MAX);
    let mut r = [0.0f32; 2];
    Fallback.gauss_noise(&mut s, &mut r).unwrap();
    assert_eq!(s.index(), 1, "the index wraps");
    let mut m = Model::at(SEED, STREAM, u64::MAX, 1);
    assert_eq!(r[0].to_bits(), f32_of(m.next_u32(), 1.0).to_bits());
    let mut whole = [0.0f32; 1];
    Fallback.gauss_noise(&mut state(1.0), &mut whole).unwrap();
    assert_eq!(r[1].to_bits(), whole[0].to_bits(), "draw 2^64 is draw 0");

    // Complex: sample u64::MAX equals sample u64::MAX - 2^63 (period 2^64 draws).
    let mut s = state(1.0);
    s.seek(u64::MAX);
    let mut a = [C32::default(); 2];
    Fallback.gauss_noise(&mut s, &mut a).unwrap();
    let mut s = state(1.0);
    s.seek(u64::MAX - (1 << 63));
    let mut b = [C32::default(); 2];
    Fallback.gauss_noise(&mut s, &mut b).unwrap();
    assert_eq!(bits_c(&a), bits_c(&b));
    let mut m = Model::at(SEED, STREAM, u64::MAX, 2);
    assert_eq!(
        a[0],
        C32::new(f32_of(m.next_u32(), 1.0), f32_of(m.next_u32(), 1.0))
    );
    let mut cq = [CQ15::default(); 1];
    let mut s = state(0.5);
    s.seek(u64::MAX);
    Fallback.gauss_noise(&mut s, &mut cq).unwrap();
    let mut m = Model::at(SEED, STREAM, u64::MAX, 2);
    assert_eq!(cq[0].re, q15_of(m.next_u32(), 0.5));
    assert_eq!(cq[0].im, q15_of(m.next_u32(), 0.5));
}

// ------------------------------------------------------------------ timing, traits

#[test]
fn timing_is_unit_ratio_without_delay() {
    let s = state(1.0);
    assert_eq!(s.ratio(), Ratio::ONE);
    assert_eq!(s.group_delay(), Delay::ZERO);
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GaussNoiseState>();
    assert_send_sync::<sdrk_core::kernels::noise::GaussScale>();
    assert_send_sync::<sdrk_core::kernels::noise::Pcg32>();
}

// ------------------------------------------------------------------ against the reference

#[cfg(feature = "reference")]
mod against_the_reference {
    use super::*;
    use sdrk_core::{Reference, Sample};

    /// Runs Fallback and Reference over the same calls (sizes, a seek, sigma changes) and
    /// compares the outputs bit for bit and the states.
    fn compare<T: Sample + Default>(sigma: f32, bits: impl Fn(&T) -> u64)
    where
        Fallback: GaussNoise<T>,
        Reference: GaussNoise<T>,
    {
        let mut a = state(sigma);
        let mut b = state(sigma);
        let mut x = vec![T::default(); 4096];
        let mut y = vec![T::default(); 4096];
        for (k, n) in [4096usize, 1, 0, 255, 64, 33, 77, 3]
            .into_iter()
            .enumerate()
        {
            let change = match k {
                2 => {
                    a.seek(123_456_789);
                    b.seek(123_456_789);
                    None
                }
                3 => Some(0.0),
                4 => Some(16.0),
                5 => Some(9e-31),
                6 => Some(1e-12),
                7 => Some(1e-30),
                _ => None,
            };
            if let Some(v) = change {
                a.set_sigma(v).unwrap();
                b.set_sigma(v).unwrap();
            }
            Fallback.gauss_noise(&mut a, &mut x[..n]).unwrap();
            Reference.gauss_noise(&mut b, &mut y[..n]).unwrap();
            let bx: Vec<u64> = x[..n].iter().map(&bits).collect();
            let by: Vec<u64> = y[..n].iter().map(&bits).collect();
            assert_eq!(bx, by, "sigma {sigma} call {k}");
            assert_eq!(a, b, "state after call {k}");
        }
    }

    #[test]
    fn fallback_equals_reference_for_every_type() {
        for sigma in [1.0f32, 0.25, 0.6, 2e-5, 16.0] {
            compare::<C32>(sigma, |c| {
                u64::from(c.re.to_bits()) << 32 | u64::from(c.im.to_bits())
            });
            compare::<CQ15>(sigma, |c| {
                u64::from(c.re.0 as u16) << 16 | u64::from(c.im.0 as u16)
            });
            compare::<f32>(sigma, |x| u64::from(x.to_bits()));
            compare::<Q15>(sigma, |x| u64::from(x.0 as u16));
        }
    }

    #[test]
    fn the_reference_refuses_a_type_change() {
        let mut s = state(1.0);
        Reference.gauss_noise(&mut s, &mut [Q15(0); 2]).unwrap();
        assert!(matches!(
            Reference.gauss_noise(&mut s, &mut [0.0f32; 2]),
            Err(Error::InvalidState(_))
        ));
    }
}
