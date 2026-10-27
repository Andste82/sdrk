//! `biquad` of `C32` and `CQ15` and `biquad_real` of `f32` and `Q15` against the fallback:
//! bit-exact output and history at every length from 0 to 300 (all tails of the blocks of
//! sixteen steps and of the flush schedule), at every start offset within a vector, for one to
//! five sections (pairs of sections in the wavefront and a single last section), for stable
//! and for unstable and resonant coefficients (overflow to infinity and NaN in `f32`,
//! saturation in `Q15`), with zeros, NaN, infinities and subnormal samples, in random chunks
//! (the phase of the flush schedule carries over), after `reset` and `set_coefs`, and for the
//! corners of the `Q15` coefficient range. The `f32` and `C32` paths of both are also compared
//! with an independent per-sample model of the flush schedule on input that reaches the flush
//! range, for up to 32 sections and in calls of 0, 1, 15, 16, 17 and random lengths. The rounding of the `Q15` paths (a vector floor or
//! the round-to-nearest-even of a shifted value, depending on the level) is swept over every
//! input value on and next to the rounding ties, for the single section and for the second
//! section of a pair. The tests hold in every build: the kernels use no multiply-add.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]

mod common;

use common::{Rng, c32_noise, cq, cq15_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::biquad::{Biquad, BiquadReal, BiquadRealState, BiquadState};
use sdrk_core::{C32, CQ15, Fallback, Q15};

const MAX_LEN: usize = 300;
const OFFSETS: [usize; 4] = [0, 1, 2, 3];

/// Equal bits, or both NaN (the payload of a NaN result is not specified).
fn same_f32(want: &[f32], got: &[f32]) -> bool {
    want.len() == got.len()
        && want
            .iter()
            .zip(got)
            .all(|(w, g)| w.to_bits() == g.to_bits() || (w.is_nan() && g.is_nan()))
}

/// Equal bits, or both NaN (the payload of a NaN result is not specified).
fn same_c32(want: &[C32], got: &[C32]) -> bool {
    let bits = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    want.len() == got.len()
        && want
            .iter()
            .zip(got)
            .all(|(w, g)| bits(w.re, g.re) && bits(w.im, g.im))
}

/// A stable section: the second-order low-pass at `w` (fraction of the sample rate) with
/// damping `d`.
fn stable(w: f32, d: f32) -> [f32; 5] {
    let k = (w * core::f32::consts::PI).tan();
    let n = 1.0 / (1.0 + d * k + k * k);
    [
        k * k * n,
        2.0 * k * k * n,
        k * k * n,
        2.0 * (k * k - 1.0) * n,
        (1.0 - d * k + k * k) * n,
    ]
}

fn coefs_c32(rng: &mut Rng, sections: usize, wild: bool) -> Vec<[f32; 5]> {
    (0..sections)
        .map(|_| {
            if wild {
                [(); 5].map(|()| rng.unit() * 2.0)
            } else {
                stable(
                    0.02 + 0.2 * (rng.unit() * 0.5 + 0.5),
                    0.3 + rng.unit().abs(),
                )
            }
        })
        .collect()
}

/// `mode` 0: the corners of the range, 1: any `i16`, 2: stable Q2.14 low-passes.
fn coefs_cq15(rng: &mut Rng, sections: usize, mode: usize) -> Vec<[i16; 5]> {
    const EDGE: [i16; 5] = [i16::MIN, i16::MAX, 0, 1, -1];
    (0..sections)
        .map(|_| match mode {
            0 => [(); 5].map(|()| EDGE[rng.below(EDGE.len())]),
            1 => [(); 5].map(|()| rng.next_u32() as i16),
            _ => stable(0.02 + 0.2 * (rng.unit() * 0.5 + 0.5), 1.4)
                .map(|v| (v * 16384.0).round() as i16),
        })
        .collect()
}

fn check_c32(coefs: &[[f32; 5]], x: &[C32], what: &str) {
    let n = coefs.len();
    let new = || BiquadState::<C32, _, _>::new(coefs.to_vec(), vec![C32::default(); 4 * n]);
    let (mut sw, mut sg) = (new().unwrap(), new().unwrap());
    let (mut want, mut got) = (vec![C32::default(); x.len()], vec![C32::default(); x.len()]);
    Fallback.biquad(&mut sw, x, &mut want).unwrap();
    Simd.biquad(&mut sg, x, &mut got).unwrap();
    assert!(same_c32(&want, &got), "{what}: output differs");
    assert!(
        same_c32(sw.history(), sg.history()),
        "{what}: history differs"
    );
}

fn check_cq15(coefs: &[[i16; 5]], x: &[CQ15], what: &str) {
    let n = coefs.len();
    let new = || BiquadState::<CQ15, _, _>::new(coefs.to_vec(), vec![CQ15::default(); 4 * n]);
    let (mut sw, mut sg) = (new().unwrap(), new().unwrap());
    let (mut want, mut got) = (
        vec![CQ15::default(); x.len()],
        vec![CQ15::default(); x.len()],
    );
    Fallback.biquad(&mut sw, x, &mut want).unwrap();
    Simd.biquad(&mut sg, x, &mut got).unwrap();
    assert_eq!(want, got, "{what}: output differs");
    assert_eq!(sw.history(), sg.history(), "{what}: history differs");
}

fn check_f32(coefs: &[[f32; 5]], x: &[f32], what: &str) {
    let n = coefs.len();
    let new = || BiquadRealState::<f32, _, _>::new(coefs.to_vec(), vec![0.0f32; 4 * n]);
    let (mut sw, mut sg) = (new().unwrap(), new().unwrap());
    let (mut want, mut got) = (vec![0.0f32; x.len()], vec![0.0f32; x.len()]);
    Fallback.biquad_real(&mut sw, x, &mut want).unwrap();
    Simd.biquad_real(&mut sg, x, &mut got).unwrap();
    assert!(same_f32(&want, &got), "{what}: output differs");
    assert!(
        same_f32(sw.history(), sg.history()),
        "{what}: history differs"
    );
    assert_eq!(sw.phase(), sg.phase(), "{what}: phase differs");
}

fn check_q15(coefs: &[[i16; 5]], x: &[Q15], what: &str) {
    let n = coefs.len();
    let new = || BiquadRealState::<Q15, _, _>::new(coefs.to_vec(), vec![Q15(0); 4 * n]);
    let (mut sw, mut sg) = (new().unwrap(), new().unwrap());
    let (mut want, mut got) = (vec![Q15(0); x.len()], vec![Q15(0); x.len()]);
    Fallback.biquad_real(&mut sw, x, &mut want).unwrap();
    Simd.biquad_real(&mut sg, x, &mut got).unwrap();
    assert_eq!(want, got, "{what}: output differs");
    assert_eq!(sw.history(), sg.history(), "{what}: history differs");
}

fn re_f32(x: &[C32]) -> Vec<f32> {
    x.iter().flat_map(|c| [c.re, c.im]).collect()
}

fn re_q15(x: &[CQ15]) -> Vec<Q15> {
    x.iter().flat_map(|c| [c.re, c.im]).collect()
}

#[test]
fn c32_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(2201);
    let x = c32_noise(&mut rng, MAX_LEN + OFFSETS.len(), 0.5);
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        let sections = 1 + n % 5;
        let coefs = coefs_c32(&mut rng, sections, false);
        check_c32(
            &coefs,
            &x[off..off + n],
            &format!("n={n} offset={off} sections={sections}"),
        );
    }
}

#[test]
fn c32_is_bit_exact_for_wild_coefficients() {
    let mut rng = Rng::new(2202);
    let x = c32_noise(&mut rng, 600, 1.0);
    for round in 0..40 {
        let sections = 1 + round % 4;
        let coefs = coefs_c32(&mut rng, sections, true);
        check_c32(&coefs, &x[..(round * 15) % 600], &format!("round {round}"));
        check_c32(&coefs, &x, &format!("round {round} full"));
    }
}

#[test]
fn c32_is_bit_exact_with_zeros_nan_infinity_and_subnormals() {
    let mut rng = Rng::new(2203);
    let base = c32_noise(&mut rng, 301, 0.5);
    let specials = [
        C32::new(0.0, 0.0),
        C32::new(-0.0, 0.0),
        C32::new(f32::NAN, 0.1),
        C32::new(0.1, f32::NAN),
        C32::new(f32::INFINITY, 0.0),
        C32::new(0.0, f32::NEG_INFINITY),
        C32::new(1e-40, 0.0),
        C32::new(1e-45, -1e-45),
        C32::new(1e-31, 1e-29),
        C32::new(3e19, 3e19),
        C32::new(f32::MAX, f32::MAX),
    ];
    for &special in &specials {
        let patterns: [&dyn Fn(usize) -> bool; 4] = [
            &|i| i == 100,
            &|i| i == 300,
            &|i| (64..128).contains(&i),
            &|i| i % 4 == 1,
        ];
        for (p, pattern) in patterns.iter().enumerate() {
            let x: Vec<C32> = base
                .iter()
                .enumerate()
                .map(|(i, &s)| if pattern(i) { special } else { s })
                .collect();
            for sections in [1, 2, 3] {
                let coefs = coefs_c32(&mut rng, sections, false);
                check_c32(
                    &coefs,
                    &x,
                    &format!("{special:?} pattern {p} sections {sections}"),
                );
            }
        }
    }
    let coefs = coefs_c32(&mut rng, 2, false);
    check_c32(&coefs, &[C32::new(0.0, 0.0); 200], "silence");
    check_c32(&coefs, &[C32::new(1e-33, -1e-33); 200], "faint");
}

#[test]
fn c32_is_bit_exact_in_random_chunks() {
    let mut rng = Rng::new(2204);
    let x = c32_noise(&mut rng, 3000, 0.5);
    for sections in 1..=4 {
        let coefs = coefs_c32(&mut rng, sections, false);
        let new = || {
            BiquadState::<C32, _, _>::new(coefs.clone(), vec![C32::default(); 4 * sections])
                .unwrap()
        };
        let (mut sw, mut sg) = (new(), new());
        let (mut want, mut got) = (vec![C32::default(); x.len()], vec![C32::default(); x.len()]);
        Fallback.biquad(&mut sw, &x, &mut want).unwrap();
        let mut done = 0;
        while done < x.len() {
            let c = (1 + rng.below(70)).min(x.len() - done);
            Simd.biquad(&mut sg, &x[done..done + c], &mut got[done..done + c])
                .unwrap();
            done += c;
        }
        assert!(same_c32(&want, &got));
        assert!(same_c32(sw.history(), sg.history()));
    }
}

#[test]
fn cq15_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(2205);
    let x = cq15_noise(&mut rng, MAX_LEN + OFFSETS.len());
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        let sections = 1 + n % 5;
        let coefs = coefs_cq15(&mut rng, sections, 2);
        check_cq15(
            &coefs,
            &x[off..off + n],
            &format!("n={n} offset={off} sections={sections}"),
        );
    }
}

#[test]
fn cq15_is_bit_exact_for_every_coefficient_range() {
    let mut rng = Rng::new(2206);
    let noise = cq15_noise(&mut rng, 700);
    let corners: Vec<CQ15> = (0..700)
        .map(|i| match i % 4 {
            0 => cq(i16::MAX, i16::MIN),
            1 => cq(i16::MIN, i16::MIN),
            2 => cq(i16::MAX, i16::MAX),
            _ => cq(0, if i % 3 == 0 { 1 } else { -1 }),
        })
        .collect();
    for mode in 0..3 {
        for round in 0..30 {
            let sections = 1 + round % 4;
            let coefs = coefs_cq15(&mut rng, sections, mode);
            check_cq15(&coefs, &noise, &format!("mode {mode} round {round} noise"));
            check_cq15(
                &coefs,
                &corners,
                &format!("mode {mode} round {round} corners"),
            );
            check_cq15(
                &coefs,
                &noise[..(round * 23) % 700],
                &format!("mode {mode} round {round} short"),
            );
        }
    }
}

#[test]
fn cq15_is_bit_exact_on_silence_dc_and_the_largest_terms() {
    let mut rng = Rng::new(2207);
    let coefs = coefs_cq15(&mut rng, 2, 2);
    check_cq15(&coefs, &[CQ15::default(); 300], "silence");
    check_cq15(&coefs, &[cq(20000, -20000); 300], "dc");
    check_cq15(&coefs, &[cq(-32768, -32768); 300], "negative full scale");
    // Five products of the largest magnitude add up to the largest accumulator.
    let largest = [[i16::MIN, i16::MIN, i16::MIN, i16::MAX, i16::MAX]];
    let x: Vec<CQ15> = (0..200)
        .map(|i| cq(if i % 2 == 0 { i16::MIN } else { i16::MAX }, i16::MIN))
        .collect();
    check_cq15(&largest, &x, "largest terms");
}

#[test]
fn cq15_is_bit_exact_in_random_chunks() {
    let mut rng = Rng::new(2208);
    let x = cq15_noise(&mut rng, 3000);
    for sections in 1..=4 {
        let coefs = coefs_cq15(&mut rng, sections, 2);
        let new = || {
            BiquadState::<CQ15, _, _>::new(coefs.clone(), vec![CQ15::default(); 4 * sections])
                .unwrap()
        };
        let (mut sw, mut sg) = (new(), new());
        let (mut want, mut got) = (
            vec![CQ15::default(); x.len()],
            vec![CQ15::default(); x.len()],
        );
        Fallback.biquad(&mut sw, &x, &mut want).unwrap();
        let mut done = 0;
        while done < x.len() {
            let c = (1 + rng.below(70)).min(x.len() - done);
            Simd.biquad(&mut sg, &x[done..done + c], &mut got[done..done + c])
                .unwrap();
            done += c;
        }
        assert_eq!(want, got);
        assert_eq!(sw.history(), sg.history());
    }
}

/// The rounding and saturation of the `f64` arithmetic against the integers: a tap with
/// `y = round_shift(b0 x, 14)` for every input value and a set of `b0` that puts the products on
/// and next to every rounding tie (`b0 x = 2^13 mod 2^14`) and beyond the saturation limits.
/// The tap runs as a single `CQ15` section (re and im together, the im part runs through the
/// values in the opposite order), as the second and as the first section of a `CQ15` pair and
/// of a `Q15` pair (the other section of the pair is the identity `b0 = 1.0`, which is exact).
/// `cq15_sweeps_every_coefficient_of_one_tap` (ignored, nightly) runs all 65536 values of `b0`.
fn sweep_one_tap(b0_values: impl Iterator<Item = i16>) {
    const ONE: [i16; 5] = [16384, 0, 0, 0, 0];
    let x: Vec<CQ15> = (0..=u16::MAX)
        .map(|v| cq(v as i16, (u16::MAX - v) as i16))
        .collect();
    let xr = re_q15(&x);
    for b0 in b0_values {
        let tap = [b0, 0, 0, 0, 0];
        check_cq15(&[tap], &x, &format!("b0 = {b0}"));
        check_cq15(&[ONE, tap], &x, &format!("pair, second, b0 = {b0}"));
        check_cq15(&[tap, ONE], &x, &format!("pair, first, b0 = {b0}"));
        check_q15(&[ONE, tap], &xr, &format!("q15 pair, second, b0 = {b0}"));
        check_q15(&[tap, ONE], &xr, &format!("q15 pair, first, b0 = {b0}"));
        check_q15(&[tap], &xr, &format!("q15 single, b0 = {b0}"));
    }
}

#[test]
fn cq15_sweeps_the_rounding_ties_and_the_saturation_of_one_tap() {
    let mut b0s: Vec<i16> = vec![
        i16::MIN,
        i16::MIN + 1,
        -16385,
        -16384,
        -16383,
        -1,
        0,
        1,
        2,
        3,
    ];
    b0s.extend([
        8191,
        8192,
        8193,
        16383,
        16384,
        16385,
        24575,
        i16::MAX - 1,
        i16::MAX,
    ]);
    for shift in 0..15 {
        let p = 1_i16 << shift;
        b0s.extend([p, p - 1, p + 1, -p]);
    }
    b0s.sort_unstable();
    b0s.dedup();
    sweep_one_tap(b0s.into_iter());
}

/// The part `k` of `n` of the exhaustive test (`SDRK_SWEEP_PART=k/n`, default `0/1`, all of
/// it): the coefficients whose index is `k` modulo `n`; `SDRK_SWEEP_STRIDE=s` (default 1) takes
/// only every `s`-th of them, for the emulators.
fn sweep_part() -> (usize, usize) {
    let Ok(part) = std::env::var("SDRK_SWEEP_PART") else {
        return (0, 1);
    };
    let parsed = part
        .split_once('/')
        .and_then(|(k, n)| Some((k.parse::<usize>().ok()?, n.parse::<usize>().ok()?)))
        .filter(|&(k, n)| k < n);
    parsed.unwrap_or_else(|| panic!("SDRK_SWEEP_PART must be k/n with k < n, not {part:?}"))
}

/// About 4.3 * 10^9 samples; run in release:
/// `cargo test -p sdrk-backend-simd --release --test biquad -- --ignored`.
#[test]
#[ignore = "long; run in release"]
fn cq15_sweeps_every_coefficient_of_one_tap() {
    let (part, parts) = sweep_part();
    let stride = std::env::var("SDRK_SWEEP_STRIDE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(1);
    sweep_one_tap((i16::MIN..=i16::MAX).skip(part).step_by(parts * stride));
}

#[test]
fn f32_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(2211);
    let x = re_f32(&c32_noise(&mut rng, MAX_LEN + OFFSETS.len(), 0.5));
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        let sections = 1 + n % 5;
        let coefs = coefs_c32(&mut rng, sections, false);
        check_f32(
            &coefs,
            &x[off..off + n],
            &format!("n={n} offset={off} sections={sections}"),
        );
    }
}

#[test]
fn f32_is_bit_exact_for_wild_coefficients() {
    let mut rng = Rng::new(2212);
    let x = re_f32(&c32_noise(&mut rng, 300, 1.0));
    for round in 0..40 {
        let sections = 1 + round % 5;
        let coefs = coefs_c32(&mut rng, sections, true);
        check_f32(&coefs, &x[..(round * 15) % 600], &format!("round {round}"));
        check_f32(&coefs, &x, &format!("round {round} full"));
    }
}

#[test]
fn f32_is_bit_exact_with_zeros_nan_infinity_and_subnormals() {
    let mut rng = Rng::new(2213);
    let base = re_f32(&c32_noise(&mut rng, 151, 0.5));
    let specials = [
        0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e-40,
        -1e-45,
        1e-31,
        -1e-29,
        3e19,
        f32::MAX,
    ];
    for &special in &specials {
        let patterns: [&dyn Fn(usize) -> bool; 4] = [
            &|i| i == 100,
            &|i| i == 301,
            &|i| (64..128).contains(&i),
            &|i| i % 4 == 1,
        ];
        for (p, pattern) in patterns.iter().enumerate() {
            let x: Vec<f32> = base
                .iter()
                .enumerate()
                .map(|(i, &s)| if pattern(i) { special } else { s })
                .collect();
            for sections in [1, 2, 3, 4] {
                let coefs = coefs_c32(&mut rng, sections, false);
                check_f32(
                    &coefs,
                    &x,
                    &format!("{special:?} pattern {p} sections {sections}"),
                );
            }
        }
    }
    for sections in [1, 2, 3] {
        let coefs = coefs_c32(&mut rng, sections, false);
        check_f32(&coefs, &[0.0; 200], "silence");
        check_f32(&coefs, &[-0.0; 200], "negative zeros");
        check_f32(&coefs, &[1e-33; 200], "faint");
        let decay: Vec<f32> = (0..400).map(|i| if i < 10 { 1.0 } else { 0.0 }).collect();
        check_f32(&coefs, &decay, "decay into the flush range");
    }
}

/// Random chunks, then `reset` and `set_coefs` in the middle of a stream: the phase of the
/// flush schedule is part of the state.
#[test]
fn f32_and_c32_are_bit_exact_in_random_chunks_and_after_reset_and_retuning() {
    let mut rng = Rng::new(2214);
    let xc = c32_noise(&mut rng, 3000, 0.5);
    let x = re_f32(&xc);
    for sections in 1..=5 {
        let coefs = coefs_c32(&mut rng, sections, false);
        let other = coefs_c32(&mut rng, sections, false);
        let new = || {
            BiquadRealState::<f32, _, _>::new(coefs.clone(), vec![0.0f32; 4 * sections]).unwrap()
        };
        let newc = || {
            BiquadState::<C32, _, _>::new(coefs.clone(), vec![C32::default(); 4 * sections])
                .unwrap()
        };
        let (mut sw, mut sg) = (new(), new());
        let (mut cw, mut cg) = (newc(), newc());
        let (mut want, mut got) = (vec![0.0f32; xc.len()], vec![0.0f32; xc.len()]);
        let (mut cwant, mut cgot) = (
            vec![C32::default(); xc.len()],
            vec![C32::default(); xc.len()],
        );
        let mut done = 0;
        let mut round = 0;
        while done < xc.len() {
            let c = (1 + rng.below(70)).min(xc.len() - done);
            let r = done..done + c;
            Fallback
                .biquad_real(&mut sw, &x[r.clone()], &mut want[r.clone()])
                .unwrap();
            Simd.biquad_real(&mut sg, &x[r.clone()], &mut got[r.clone()])
                .unwrap();
            Fallback
                .biquad(&mut cw, &xc[r.clone()], &mut cwant[r.clone()])
                .unwrap();
            Simd.biquad(&mut cg, &xc[r.clone()], &mut cgot[r]).unwrap();
            assert_eq!(sw.phase(), sg.phase());
            assert_eq!(cw.phase(), cg.phase());
            match round % 7 {
                3 => {
                    sw.set_coefs(other.clone()).unwrap();
                    sg.set_coefs(other.clone()).unwrap();
                    cw.set_coefs(other.clone()).unwrap();
                    cg.set_coefs(other.clone()).unwrap();
                }
                5 => {
                    sw.reset();
                    sg.reset();
                    cw.reset();
                    cg.reset();
                    assert_eq!(sg.phase(), 0);
                }
                _ => {}
            }
            done += c;
            round += 1;
        }
        assert!(same_f32(&want, &got), "sections {sections}");
        assert!(same_f32(sw.history(), sg.history()));
        assert!(same_c32(&cwant, &cgot), "c32 sections {sections}");
        assert!(same_c32(cw.history(), cg.history()));
    }
}

#[test]
fn q15_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(2215);
    let x = re_q15(&cq15_noise(&mut rng, MAX_LEN + OFFSETS.len()));
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        let sections = 1 + n % 5;
        let coefs = coefs_cq15(&mut rng, sections, 2);
        check_q15(
            &coefs,
            &x[off..off + n],
            &format!("n={n} offset={off} sections={sections}"),
        );
    }
}

#[test]
fn q15_is_bit_exact_for_every_coefficient_range() {
    let mut rng = Rng::new(2216);
    let noise = re_q15(&cq15_noise(&mut rng, 350));
    let corners: Vec<Q15> = (0..700)
        .map(|i| match i % 5 {
            0 | 3 => Q15(i16::MAX),
            1 | 2 => Q15(i16::MIN),
            _ => Q15(if i % 3 == 0 { 1 } else { -1 }),
        })
        .collect();
    for mode in 0..3 {
        for round in 0..30 {
            let sections = 1 + round % 5;
            let coefs = coefs_cq15(&mut rng, sections, mode);
            check_q15(&coefs, &noise, &format!("mode {mode} round {round} noise"));
            check_q15(
                &coefs,
                &corners,
                &format!("mode {mode} round {round} corners"),
            );
            check_q15(
                &coefs,
                &noise[..(round * 23) % 700],
                &format!("mode {mode} round {round} short"),
            );
        }
    }
}

#[test]
fn q15_is_bit_exact_on_silence_dc_and_the_largest_terms() {
    let mut rng = Rng::new(2217);
    for sections in [1, 2, 3] {
        let coefs = coefs_cq15(&mut rng, sections, 2);
        check_q15(&coefs, &[Q15(0); 300], "silence");
        check_q15(&coefs, &[Q15(20000); 300], "dc");
        check_q15(&coefs, &[Q15(-32768); 300], "negative full scale");
    }
    // Five products of the largest magnitude add up to the largest accumulator, in both
    // sections of a pair.
    let largest = [i16::MIN, i16::MIN, i16::MIN, i16::MAX, i16::MAX];
    let x: Vec<Q15> = (0..200)
        .map(|i| Q15(if i % 2 == 0 { i16::MIN } else { i16::MAX }))
        .collect();
    check_q15(&[largest], &x, "largest terms");
    check_q15(&[largest, largest], &x, "largest terms, pair");
    let xc: Vec<CQ15> = x.iter().map(|v| cq(v.0, i16::MIN)).collect();
    check_cq15(&[largest, largest], &xc, "largest terms, cq15 pair");
}

#[test]
fn q15_is_bit_exact_in_random_chunks() {
    let mut rng = Rng::new(2218);
    let x = re_q15(&cq15_noise(&mut rng, 1500));
    for sections in 1..=5 {
        let coefs = coefs_cq15(&mut rng, sections, 2);
        let new = || {
            BiquadRealState::<Q15, _, _>::new(coefs.clone(), vec![Q15(0); 4 * sections]).unwrap()
        };
        let (mut sw, mut sg) = (new(), new());
        let (mut want, mut got) = (vec![Q15(0); x.len()], vec![Q15(0); x.len()]);
        Fallback.biquad_real(&mut sw, &x, &mut want).unwrap();
        let mut done = 0;
        while done < x.len() {
            let c = (1 + rng.below(70)).min(x.len() - done);
            Simd.biquad_real(&mut sg, &x[done..done + c], &mut got[done..done + c])
                .unwrap();
            done += c;
        }
        assert_eq!(want, got);
        assert_eq!(sw.history(), sg.history());
    }
}

/// An independent per-sample model of one component of the `f32` cascade and its flush
/// schedule (decision D17): every sample runs through all sections in turn, and section `k`
/// sets each of its two states below 1e-30 in magnitude to zero after sample `n` when
/// `(n + k) % 16 == 15`, `n` counted since the creation or the last `reset`. It shares no
/// code with the kernels; `changed` counts the flushes that changed a value, so that a test
/// can show that it reached the flush range.
struct FlushModel {
    coefs: Vec<[f32; 5]>,
    s: Vec<[f32; 2]>,
    n: usize,
    changed: usize,
}

impl FlushModel {
    fn new(coefs: &[[f32; 5]]) -> Self {
        FlushModel {
            coefs: coefs.to_vec(),
            s: vec![[0.0; 2]; coefs.len()],
            n: 0,
            changed: 0,
        }
    }

    fn reset(&mut self) {
        self.s.fill([0.0; 2]);
        self.n = 0;
    }

    fn step(&mut self, x: f32) -> f32 {
        let mut v = x;
        for (k, (c, s)) in self.coefs.iter().zip(&mut self.s).enumerate() {
            let y = c[0] * v + s[0];
            s[0] = (c[1] * v + s[1]) - c[3] * y;
            s[1] = c[2] * v - c[4] * y;
            if (self.n + k) % 16 == 15 {
                for e in s.iter_mut() {
                    if e.abs() < 1e-30 {
                        if *e != 0.0 {
                            self.changed += 1;
                        }
                        *e = 0.0;
                    }
                }
            }
            v = y;
        }
        self.n += 1;
        v
    }
}

/// Input that drives the states into the flush range (0, 1e-30): bursts of noise of the
/// amplitude 1e-28 and impulses of 1e-20 and 1, each followed by a silence.
fn flush_active(rng: &mut Rng, n: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        match rng.below(4) {
            0 | 1 => {
                let len = 1 + rng.below(60);
                for v in x.iter_mut().skip(i).take(len) {
                    *v = rng.unit() * 1e-28;
                }
                i += len;
            }
            2 => {
                x[i] = 1e-20;
                i += 1;
            }
            _ => {
                x[i] = 1.0;
                i += 1;
            }
        }
        i += 40 + rng.below(300);
    }
    x
}

/// Splits `len` samples into calls of the lengths 0, 1, 15, 16 and 17 in turn with random
/// lengths of 0 to 70.
fn call_lengths(rng: &mut Rng, len: usize) -> Vec<usize> {
    const FIXED: [usize; 5] = [0, 1, 15, 16, 17];
    let mut out = Vec::new();
    let mut left = len;
    let mut i = 0;
    while left > 0 {
        let c = if i % 2 == 0 {
            FIXED[(i / 2) % FIXED.len()]
        } else {
            rng.below(71)
        };
        let c = c.min(left);
        out.push(c);
        left -= c;
        i += 1;
    }
    out
}

/// What happens after a call of a stream: every eleventh call retunes (alternating between
/// the two sets of coefficients) or resets.
fn event(call: usize) -> Option<bool> {
    match call % 11 {
        4 => Some(true),
        9 => Some(false),
        _ => None,
    }
}

/// Runs `x` through `backend` in calls of `lengths` and through the model, with the events of
/// [`event`], and compares the output, the states and the phase bit for bit after every call.
/// Returns the number of flushes that changed a value.
fn model_f32<B: BiquadReal<f32>>(
    backend: &B,
    what: &str,
    sets: [&[[f32; 5]]; 2],
    x: &[f32],
    lengths: &[usize],
) -> usize {
    let n = sets[0].len();
    let mut model = FlushModel::new(sets[0]);
    let mut state =
        BiquadRealState::<f32, _, _>::new(sets[0].to_vec(), vec![0.0f32; 4 * n]).unwrap();
    let mut out = vec![0.0f32; x.len()];
    let (mut done, mut set) = (0, 0);
    for (call, &c) in lengths.iter().enumerate() {
        let r = done..done + c;
        backend
            .biquad_real(&mut state, &x[r.clone()], &mut out[r.clone()])
            .unwrap();
        for i in r {
            let want = model.step(x[i]);
            assert!(
                want.to_bits() == out[i].to_bits(),
                "{what}, {n} sections, sample {i}: want {want:e}, got {:e}",
                out[i]
            );
        }
        assert_eq!(
            state.phase(),
            model.n % 16,
            "{what}, {n} sections, call {call}"
        );
        for (k, s) in model.s.iter().enumerate() {
            let h = state.history();
            assert!(
                h[4 * k].to_bits() == s[0].to_bits() && h[4 * k + 1].to_bits() == s[1].to_bits(),
                "{what}, {n} sections, call {call}: section {k} has {:?}, the model {s:?}",
                &h[4 * k..4 * k + 2]
            );
        }
        match event(call) {
            Some(true) => {
                set ^= 1;
                state.set_coefs(sets[set].to_vec()).unwrap();
                model.coefs = sets[set].to_vec();
            }
            Some(false) => {
                state.reset();
                model.reset();
            }
            None => {}
        }
        done += c;
    }
    model.changed
}

/// [`model_f32`] for `C32`: the model filters re and im as two components.
fn model_c32<B: Biquad<C32>>(
    backend: &B,
    what: &str,
    sets: [&[[f32; 5]]; 2],
    x: &[C32],
    lengths: &[usize],
) -> usize {
    let n = sets[0].len();
    let (mut re, mut im) = (FlushModel::new(sets[0]), FlushModel::new(sets[0]));
    let mut state =
        BiquadState::<C32, _, _>::new(sets[0].to_vec(), vec![C32::default(); 4 * n]).unwrap();
    let mut out = vec![C32::default(); x.len()];
    let (mut done, mut set) = (0, 0);
    for (call, &c) in lengths.iter().enumerate() {
        let r = done..done + c;
        backend
            .biquad(&mut state, &x[r.clone()], &mut out[r.clone()])
            .unwrap();
        for i in r {
            let want = C32::new(re.step(x[i].re), im.step(x[i].im));
            assert!(
                same_c32(&[want], &out[i..=i]),
                "{what}, {n} sections, sample {i}: want {want:?}, got {:?}",
                out[i]
            );
        }
        assert_eq!(
            state.phase(),
            re.n % 16,
            "{what}, {n} sections, call {call}"
        );
        for k in 0..n {
            let h = state.history();
            let want = [
                C32::new(re.s[k][0], im.s[k][0]),
                C32::new(re.s[k][1], im.s[k][1]),
            ];
            assert!(
                same_c32(&want, &h[4 * k..4 * k + 2]),
                "{what}, {n} sections, call {call}: section {k} has {:?}, the model {want:?}",
                &h[4 * k..4 * k + 2]
            );
        }
        match event(call) {
            Some(true) => {
                set ^= 1;
                state.set_coefs(sets[set].to_vec()).unwrap();
                re.coefs = sets[set].to_vec();
                im.coefs = sets[set].to_vec();
            }
            Some(false) => {
                state.reset();
                re.reset();
                im.reset();
            }
            None => {}
        }
        done += c;
    }
    re.changed + im.changed
}

/// The fallback and the SIMD path against the independent model of the flush schedule, on
/// input that reaches the flush range (so that a flush on the wrong sample changes the
/// output), in one call and in calls of 0, 1, 15, 16, 17 and random lengths with `reset` and
/// `set_coefs` in the middle of the stream, for 1 to 5, 31 and 32 sections.
#[test]
fn f32_and_c32_follow_an_independent_model_of_the_flush_schedule() {
    let mut rng = Rng::new(2230);
    for sections in [1, 2, 3, 4, 5, 31, 32] {
        let a = coefs_c32(&mut rng, sections, false);
        let b = coefs_c32(&mut rng, sections, false);
        let x = flush_active(&mut rng, 4000);
        let xi = flush_active(&mut rng, 4000);
        let xc: Vec<C32> = x.iter().zip(&xi).map(|(&r, &i)| C32::new(r, i)).collect();
        let lengths = call_lengths(&mut rng, x.len());
        let what = |side: &str, how: &str| format!("{side} {how}");
        for (how, lengths) in [("in one call", &[x.len()][..]), ("in chunks", &lengths[..])] {
            let changed = model_f32(&Fallback, &what("f32 fallback", how), [&a, &b], &x, lengths);
            assert!(
                changed >= 20,
                "f32, {sections} sections {how}: only {changed} flushes changed a value"
            );
            model_f32(&Simd, &what("f32 simd", how), [&a, &b], &x, lengths);
            let changed = model_c32(
                &Fallback,
                &what("c32 fallback", how),
                [&a, &b],
                &xc,
                lengths,
            );
            assert!(
                changed >= 40,
                "c32, {sections} sections {how}: only {changed} flushes changed a value"
            );
            model_c32(&Simd, &what("c32 simd", how), [&a, &b], &xc, lengths);
        }
    }
}
