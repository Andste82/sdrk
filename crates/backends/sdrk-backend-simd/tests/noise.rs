//! `gauss_noise` (M48 test 9) against the fallback, bit for bit and state for state, at the level
//! of the running build: every length from 0 to 67 and 4096 at the start indices 0, 1 and
//! `2^32 + 3` for all four sample types and every `sigma` of the unit tests (including the
//! flushed, the zero and the `sh ≥ 45` scales and 16, where `Q15` saturates); draws placed by
//! construction in every octave `L = 0 ..= 31` with both signs at every lane position of the
//! vector blocks; chunk plans with a `sigma` change; the state rules; and the golden cases of
//! M48 §4.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::many_single_char_names
)]

mod common;

use std::path::PathBuf;

use common::{Rng, chunks};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::noise::{GaussNoise, GaussNoiseState, PCG32_MULT, normal_q17};
use sdrk_core::{C32, CQ15, Error, Fallback, Q15, Sample, SampleType};

/// The seed of the generators of this file (M48 range).
const SEED: u64 = 680;
/// The seed of the chunk plans (M48 range).
const PLAN_SEED: u64 = 687;
/// Chunk plans per type.
const PLANS: u64 = 20;

const STARTS: [u64; 3] = [0, 1, (1 << 32) + 3];

/// The `sigma` values of the unit tests (M48 test 2) and the extremes of test 6.
fn sigmas() -> [f32; 12] {
    [
        0.0,
        9e-31,
        1e-30,
        1e-12,
        1.0 / 1_048_576.0,
        1e-4,
        0.25,
        0.6,
        1.0,
        4.0,
        f32::from_bits(16.0f32.to_bits() - 1),
        16.0,
    ]
}

/// The raw bits of a sample, for comparing signed zeros and every bit.
trait Bits: Sample + Copy + Default {
    fn raw(&self) -> (u32, u32);
}
impl Bits for f32 {
    fn raw(&self) -> (u32, u32) {
        (self.to_bits(), 0)
    }
}
impl Bits for Q15 {
    fn raw(&self) -> (u32, u32) {
        (u32::from(self.0 as u16), 0)
    }
}
impl Bits for C32 {
    fn raw(&self) -> (u32, u32) {
        (self.re.to_bits(), self.im.to_bits())
    }
}
impl Bits for CQ15 {
    fn raw(&self) -> (u32, u32) {
        (u32::from(self.re.0 as u16), u32::from(self.im.0 as u16))
    }
}

fn raw<T: Bits>(x: &[T]) -> Vec<(u32, u32)> {
    x.iter().map(Bits::raw).collect()
}

/// One call of each backend from the same state: the same samples and the same state after.
fn same<T: Bits>(state: &GaussNoiseState, n: usize, what: &str)
where
    Fallback: GaussNoise<T>,
    Simd: GaussNoise<T>,
{
    let (mut a, mut b) = (*state, *state);
    let mut want = vec![T::default(); n];
    let mut got = vec![T::default(); n];
    Fallback.gauss_noise(&mut a, &mut want).unwrap();
    Simd.gauss_noise(&mut b, &mut got).unwrap();
    assert_eq!(raw(&want), raw(&got), "{what}");
    assert_eq!(a, b, "{what}: state");
}

fn every_length<T: Bits>(stream: u64)
where
    Fallback: GaussNoise<T>,
    Simd: GaussNoise<T>,
{
    for sigma in sigmas() {
        for start in STARTS {
            for n in (0..=67).chain([4096]) {
                let mut st = GaussNoiseState::new(SEED, stream, sigma).unwrap();
                st.seek(start);
                same::<T>(
                    &st,
                    n,
                    &format!("{:?} sigma {sigma:e} start {start} n {n}", T::TYPE),
                );
            }
        }
    }
}

#[test]
fn f32_gives_the_bits_of_the_fallback_at_every_length() {
    every_length::<f32>(1);
}

#[test]
fn q15_gives_the_bits_of_the_fallback_at_every_length() {
    every_length::<Q15>(2);
}

#[test]
fn c32_gives_the_bits_of_the_fallback_at_every_length() {
    every_length::<C32>(3);
}

#[test]
fn cq15_gives_the_bits_of_the_fallback_at_every_length() {
    every_length::<CQ15>(4);
}

// ---------------------------------------------------------------------------------------------
// Draws placed in every octave.

/// `(mult, plus)` of `delta` steps of the LCG with the increment `inc`.
fn jump(inc: u64, mut delta: u64) -> (u64, u64) {
    let (mut cm, mut cp) = (PCG32_MULT, inc);
    let (mut am, mut ap) = (1_u64, 0_u64);
    while delta > 0 {
        if delta & 1 == 1 {
            am = am.wrapping_mul(cm);
            ap = ap.wrapping_mul(cm).wrapping_add(cp);
        }
        cp = cm.wrapping_add(1).wrapping_mul(cp);
        cm = cm.wrapping_mul(cm);
        delta >>= 1;
    }
    (am, ap)
}

/// The inverse of an odd `a` modulo 2^64 (Newton's iteration).
fn inverse(a: u64) -> u64 {
    let mut x = a;
    for _ in 0..6 {
        x = x.wrapping_mul(2_u64.wrapping_sub(a.wrapping_mul(x)));
    }
    assert_eq!(a.wrapping_mul(x), 1);
    x
}

/// A 64-bit LCG state whose PCG32 output is `r`, with the rotation `rot` and the free low bits
/// `low`: the top five bits are `rot`, and bits 27 to 58 solve
/// `((s >> 18) ^ s) >> 27 = rotl(r, rot)` from the top down.
fn state_for(r: u32, rot: u32, low: u64) -> u64 {
    let xs = r.rotate_left(rot);
    let mut s = (u64::from(rot) << 59) | (low & ((1 << 27) - 1));
    for i in (27..=58).rev() {
        let above = if i + 18 < 64 { (s >> (i + 18)) & 1 } else { 0 };
        let bit = u64::from((xs >> (i - 27)) & 1) ^ above;
        s |= bit << i;
    }
    s
}

/// The seed (with the stream) whose generator has the state `s` after `d` draws.
fn seed_for(stream: u64, d: u64, s: u64) -> u64 {
    let inc = (stream << 1) | 1;
    let (a, c) = jump(inc, d);
    // State after `new` = (inc + seed) · MULT + inc.
    let s0 = s.wrapping_sub(c).wrapping_mul(inverse(a));
    s0.wrapping_sub(inc)
        .wrapping_mul(inverse(PCG32_MULT))
        .wrapping_sub(inc)
}

/// A draw in the octave `l` (of `x = 2r + 1`) with the sign `neg`.
fn draw_in_octave(rng: &mut Rng, l: u32, neg: bool) -> u32 {
    let v = if l == 0 {
        0
    } else {
        (1 << (l - 1)) | (rng.next_u32() & ((1 << (l - 1)) - 1))
    };
    v | (u32::from(neg) << 31)
}

/// Every octave and sign at the draw offsets 0 to 15 (every lane position of the blocks of 4, 8
/// and 16 lanes), 16, 21, 31 and 33 (later blocks) and 44 (the remainder of a real call) of a
/// call of `n` samples from the sample `start`: the generator is constructed so that the draw
/// at the offset is the chosen one.
fn every_octave<T: Bits>()
where
    Fallback: GaussNoise<T>,
    Simd: GaussNoise<T>,
{
    let d = match T::TYPE {
        SampleType::C32 | SampleType::CQ15 => 2,
        _ => 1,
    };
    let mut rng = Rng::new(SEED + 9);
    let n = 45;
    for l in 0..32 {
        for neg in [false, true] {
            for offset in (0_u64..16).chain([16, 21, 31, 33, 44]) {
                for (start, sigma) in [(0_u64, 1.0_f32), (1_000_003, 0.6), (77, 16.0)] {
                    let r = draw_in_octave(&mut rng, l, neg);
                    let rot = rng.next_u32() & 31;
                    let s = state_for(r, rot, u64::from(rng.next_u32()));
                    let stream = u64::from(rng.next_u32());
                    let at = start * d + offset;
                    let mut st =
                        GaussNoiseState::new(seed_for(stream, at, s), stream, sigma).unwrap();
                    st.seek(start);
                    // The construction: the draw at the offset is `r`.
                    let mut check = st;
                    let (g, _) = check.begin(T::TYPE, 1).unwrap().unwrap();
                    g.advance(offset);
                    assert_eq!(g.next_u32(), r);
                    let x = ((r << 1) | 1).leading_zeros();
                    assert_eq!(31 - x, l);
                    assert_eq!(normal_q17(r) < 0, neg && normal_q17(r) != 0);
                    same::<T>(
                        &st,
                        n,
                        &format!(
                            "{:?} L {l} neg {neg} offset {offset} start {start}",
                            T::TYPE
                        ),
                    );
                }
            }
        }
    }
}

#[test]
fn draws_in_every_octave_with_both_signs_give_the_bits_of_the_fallback() {
    every_octave::<f32>();
    every_octave::<Q15>();
    every_octave::<C32>();
    every_octave::<CQ15>();
}

// ---------------------------------------------------------------------------------------------
// Chunk plans, zeros and the state rules.

/// The whole call of the fallback against the backend in the chunks of a plan, with a
/// `sigma` change in the middle.
fn plans<T: Bits>()
where
    Fallback: GaussNoise<T>,
    Simd: GaussNoise<T>,
{
    const N: usize = 3000;
    for k in 0..PLANS {
        let mut rng = Rng::new(PLAN_SEED * 100 + k);
        let mut want = vec![T::default(); N];
        let mut got = vec![T::default(); N];
        let mut a = GaussNoiseState::new(SEED + k, k, 0.25).unwrap();
        let mut b = a;
        let half = N / 2;
        Fallback.gauss_noise(&mut a, &mut want[..half]).unwrap();
        a.set_sigma(2.0).unwrap();
        Fallback.gauss_noise(&mut a, &mut want[half..]).unwrap();
        let mut at = 0;
        for len in chunks(&mut rng, N) {
            // The change of `sigma` at the same sample: a chunk that crosses it is split.
            let end = at + len;
            if at < half && end > half {
                Simd.gauss_noise(&mut b, &mut got[at..half]).unwrap();
                b.set_sigma(2.0).unwrap();
                Simd.gauss_noise(&mut b, &mut got[half..end]).unwrap();
            } else {
                if at == half {
                    b.set_sigma(2.0).unwrap();
                }
                Simd.gauss_noise(&mut b, &mut got[at..end]).unwrap();
            }
            at = end;
        }
        assert_eq!(at, N);
        assert_eq!(raw(&want), raw(&got), "{:?} plan {k}", T::TYPE);
        assert_eq!(a, b, "{:?} plan {k}: state", T::TYPE);
    }
}

#[test]
fn chunk_plans_give_the_bits_of_one_fallback_call() {
    plans::<f32>();
    plans::<Q15>();
    plans::<C32>();
    plans::<CQ15>();
}

#[test]
fn sigma_zero_gives_positive_zeros_and_keeps_the_position() {
    for sigma in [0.0, 9e-31] {
        let mut a = GaussNoiseState::new(SEED, 9, sigma).unwrap();
        let mut b = a;
        let mut got = vec![C32::new(1.0, 1.0); 77];
        Simd.gauss_noise(&mut b, &mut got).unwrap();
        assert!(
            got.iter()
                .all(|c| c.re.to_bits() == 0 && c.im.to_bits() == 0)
        );
        let mut want = vec![C32::default(); 77];
        Fallback.gauss_noise(&mut a, &mut want).unwrap();
        assert_eq!(a, b);
        a.set_sigma(1.0).unwrap();
        b.set_sigma(1.0).unwrap();
        same::<C32>(&a, 100, "after sigma 0");
    }
}

#[test]
fn the_state_rules_are_those_of_the_fallback() {
    let mut st = GaussNoiseState::new(SEED, 5, 0.5).unwrap();
    // Empty calls change nothing.
    let before = st;
    Simd.gauss_noise(&mut st, &mut [] as &mut [f32]).unwrap();
    assert_eq!(st, before);
    Simd.gauss_noise(&mut st, &mut [0.0f32; 10]).unwrap();
    let fixed = st;
    // Another type is `InvalidState`, and nothing changes.
    assert!(matches!(
        Simd.gauss_noise(&mut st, &mut [C32::default(); 4]),
        Err(Error::InvalidState(_))
    ));
    assert!(matches!(
        Simd.gauss_noise(&mut st, &mut [] as &mut [Q15]),
        Err(Error::InvalidState(_))
    ));
    assert_eq!(st, fixed);
    // After a seek far into the sequence the type is free again.
    st.seek(u64::MAX / 8);
    same::<CQ15>(&st, 257, "seek u64::MAX / 8");
    st.seek(u64::MAX);
    same::<f32>(&st, 99, "seek u64::MAX");
    st.reset();
    same::<Q15>(&st, 33, "reset");
}

#[test]
fn a_seek_equals_the_skipped_samples() {
    let mut st = GaussNoiseState::new(SEED, 11, 1.0).unwrap();
    let mut whole = vec![C32::default(); 600];
    Simd.gauss_noise(&mut st, &mut whole).unwrap();
    for start in [0u64, 1, 31, 32, 33, 299, 599] {
        st.seek(start);
        let mut part = vec![C32::default(); 600 - start as usize];
        Simd.gauss_noise(&mut st, &mut part).unwrap();
        assert_eq!(raw(&part), raw(&whole[start as usize..]), "start {start}");
    }
}

// ---------------------------------------------------------------------------------------------
// The golden cases of M48 §4.

/// The bytes of a golden file of `gauss_noise`.
fn golden(file: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("tests")
        .join("golden")
        .join("vectors")
        .join("gauss_noise")
        .join(file);
    std::fs::read(&path).unwrap()
}

/// The case `name` through the backend: its parameters (checked against the JSON), the run
/// whole and in chunks, and the bytes of `expected`.
fn golden_case<T: Bits>(name: &str, seed: u64, stream: u64, sigma: f32, start: u64)
where
    Simd: GaussNoise<T>,
{
    let json = String::from_utf8(golden(&format!("{name}.json"))).unwrap();
    for field in [
        format!("\"generator_seed\": {seed},"),
        format!("\"stream\": {stream}"),
        format!("\"sigma_bits\": {},", sigma.to_bits()),
        format!("\"start_index\": {start},"),
        "\"len\": 4096,".to_owned(),
    ] {
        assert!(json.contains(&field), "{name}: {field}");
    }
    let want = golden(&format!("{name}.expected.bin"));
    let fresh = || {
        let mut st = GaussNoiseState::new(seed, stream, sigma).unwrap();
        st.seek(start);
        st
    };
    let mut st = fresh();
    let mut got = vec![T::default(); 4096];
    Simd.gauss_noise(&mut st, &mut got).unwrap();
    assert_eq!(st.index(), start + 4096, "{name}");
    let bytes = |x: &[T]| -> Vec<u8> {
        x.iter()
            .flat_map(|v| {
                let (a, b) = v.raw();
                let mut out = Vec::new();
                match T::TYPE {
                    SampleType::C32 => {
                        out.extend(a.to_le_bytes());
                        out.extend(b.to_le_bytes());
                    }
                    SampleType::F32 => out.extend(a.to_le_bytes()),
                    SampleType::CQ15 => {
                        out.extend((a as u16).to_le_bytes());
                        out.extend((b as u16).to_le_bytes());
                    }
                    _ => out.extend((a as u16).to_le_bytes()),
                }
                out
            })
            .collect()
    };
    assert!(bytes(&got) == want, "{name}: whole");
    let mut rng = Rng::new(PLAN_SEED + seed);
    let mut st = fresh();
    let mut parts = vec![T::default(); 4096];
    let mut at = 0;
    for len in chunks(&mut rng, 4096) {
        Simd.gauss_noise(&mut st, &mut parts[at..at + len]).unwrap();
        at += len;
    }
    assert!(bytes(&parts) == want, "{name}: chunks");
}

#[test]
fn the_golden_cases_give_their_bits() {
    golden_case::<C32>("c32_unit_4096", 680, 1, 1.0, 0);
    golden_case::<C32>("c32_seek_4096", 681, 7, 0.3, 1_000_000_007);
    golden_case::<f32>("f32_unit_4096", 682, 1, 1.0, 0);
    golden_case::<CQ15>("cq15_quarter_4096", 683, 2, 0.25, 0);
    golden_case::<Q15>("q15_clip_4096", 684, 3, 0.6, 0);
}
