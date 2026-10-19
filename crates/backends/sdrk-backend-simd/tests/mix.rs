//! `rotate` and `fm_mod` against the fallback and the reference.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

mod common;

use common::{LENGTHS, Rng, bits, c32_noise, chunks, cq15_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::kernels::modulate::{FmMod, FmModState};
use sdrk_core::{C32, CQ15, Fallback, Q15, Rate, Reference};

fn snr_db(want: &[C32], got: &[C32]) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    for (w, g) in want.iter().zip(got) {
        let (wr, wi) = (f64::from(w.re), f64::from(w.im));
        sig += wr * wr + wi * wi;
        err += (wr - f64::from(g.re)).powi(2) + (wi - f64::from(g.im)).powi(2);
    }
    if err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (sig / err).log10()
    }
}

#[test]
fn rotate_c32_meets_the_contract() {
    // 120 dB against the exact rotation (the polynomial oscillator reaches about 140 dB).
    let mut rng = Rng::new(10);
    for &n in &LENGTHS {
        for inc in [
            0u32,
            1,
            1 << 30,
            3 << 30,
            0x1234_5678,
            u32::MAX,
            0x0FFF_FFFF,
            0x8000_0000,
        ] {
            let x = c32_noise(&mut rng, n, 1.0);
            let phase = rng.next_u32();
            let mut a = Nco::from_increment(inc);
            let mut b = a;
            a.set_phase(phase);
            b.set_phase(phase);
            let (mut want, mut got) = (vec![C32::default(); n], vec![C32::default(); n]);
            Reference.rotate(&mut a, &x, &mut want).unwrap();
            Simd.rotate(&mut b, &x, &mut got).unwrap();
            if n > 0 {
                let snr = snr_db(&want, &got);
                assert!(snr > 130.0, "n={n} inc={inc:#x} snr={snr}");
            }
            assert_eq!(a, b);
        }
    }
}

#[test]
fn rotate_cq15_is_bit_exact() {
    let mut rng = Rng::new(11);
    for &n in &LENGTHS {
        for inc in [0u32, 1, 1 << 30, 3 << 30, 0x1234_5678, u32::MAX] {
            let x = cq15_noise(&mut rng, n);
            let mut a = Nco::from_increment(inc);
            a.set_phase(rng.next_u32());
            let mut b = a;
            let mut want = vec![CQ15::default(); n];
            let mut got = vec![CQ15::default(); n];
            Fallback.rotate(&mut a, &x, &mut want).unwrap();
            Simd.rotate(&mut b, &x, &mut got).unwrap();
            assert_eq!(want, got, "n={n} inc={inc:#x}");
            assert_eq!(a, b);
        }
    }
}

#[test]
fn rotate_full_scale_and_non_finite() {
    let n = 300;
    let mut x = vec![C32::new(1.0, -1.0); n];
    x[3] = C32::new(f32::NAN, 0.0);
    x[40] = C32::new(f32::INFINITY, f32::NEG_INFINITY);
    x[41] = C32::new(-0.0, 0.0);
    let mut a = Nco::from_increment(0xC000_0001);
    let mut b = a;
    let (mut want, mut got) = (vec![C32::default(); n], vec![C32::default(); n]);
    Fallback.rotate(&mut a, &x, &mut want).unwrap();
    Simd.rotate(&mut b, &x, &mut got).unwrap();
    // The samples around the non-finite ones are unaffected and close to the fallback; the
    // non-finite ones stay non-finite.
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        if i == 3 || i == 40 {
            assert!(!g.re.is_finite() || !g.im.is_finite(), "i={i}");
        } else {
            assert!(
                (w.re - g.re).abs() < 1e-5 && (w.im - g.im).abs() < 1e-5,
                "i={i} {w:?} {g:?}"
            );
        }
    }
    assert_eq!(a, b);
    let q = vec![
        CQ15 {
            re: Q15(-32768),
            im: Q15(-32768)
        };
        n
    ];
    let mut a = Nco::from_increment(0x0123_4567);
    let mut b = a;
    let (mut want, mut got) = (vec![CQ15::default(); n], vec![CQ15::default(); n]);
    Fallback.rotate(&mut a, &q, &mut want).unwrap();
    Simd.rotate(&mut b, &q, &mut got).unwrap();
    assert_eq!(want, got);
}

#[test]
fn rotate_is_block_size_invariant() {
    let mut rng = Rng::new(12);
    let n = 2000;
    let x = c32_noise(&mut rng, n, 1.0);
    let mut nco = Nco::new(123_456.7, Rate::hz(2_400_000).unwrap()).unwrap();
    let mut whole = vec![C32::default(); n];
    let start = nco;
    Simd.rotate(&mut nco, &x, &mut whole).unwrap();
    let mut nco2 = start;
    let mut parts = vec![C32::default(); n];
    let mut at = 0;
    for c in chunks(&mut rng, n) {
        Simd.rotate(&mut nco2, &x[at..at + c], &mut parts[at..at + c])
            .unwrap();
        at += c;
    }
    assert_eq!(bits(&whole), bits(&parts));
    assert_eq!(nco, nco2);
}

#[test]
fn rotate_length_mismatch_is_an_error() {
    let mut nco = Nco::from_increment(1);
    let mut out = [C32::default(); 3];
    assert_eq!(
        Fallback.rotate(&mut nco, &[C32::default(); 4], &mut out),
        Simd.rotate(&mut nco, &[C32::default(); 4], &mut out)
    );
}

fn fm_state(deviation: f64, rate: u64) -> FmModState {
    FmModState::new(deviation, Rate::hz(rate).unwrap()).unwrap()
}

#[test]
fn fm_mod_c32_is_bit_exact() {
    let mut rng = Rng::new(13);
    for &n in &LENGTHS {
        for (dev, rate) in [
            (75_000.0, 240_000),
            (1.0, 1_000_000),
            (1e6, 1_000),
            (0.0, 48_000),
        ] {
            let x: Vec<f32> = (0..n).map(|_| rng.unit() * 1.1).collect();
            let (mut a, mut b) = (fm_state(dev, rate), fm_state(dev, rate));
            a.set_phase(rng.next_u32());
            b.set_phase(a.phase());
            let (mut want, mut got) = (vec![C32::default(); n], vec![C32::default(); n]);
            Fallback.fm_mod(&mut a, &x, &mut want).unwrap();
            Simd.fm_mod(&mut b, &x, &mut got).unwrap();
            assert_eq!(bits(&want), bits(&got), "n={n} dev={dev}");
            assert_eq!(a.phase(), b.phase());
        }
    }
}

#[test]
fn fm_mod_cq15_is_bit_exact() {
    let mut rng = Rng::new(14);
    for &n in &LENGTHS {
        for (dev, rate) in [(75_000.0, 240_000), (1.0, 1_000_000), (1e6, 1_000)] {
            let x: Vec<Q15> = (0..n).map(|_| Q15(rng.next_u32() as i16)).collect();
            let (mut a, mut b) = (fm_state(dev, rate), fm_state(dev, rate));
            let (mut want, mut got) = (vec![CQ15::default(); n], vec![CQ15::default(); n]);
            Fallback.fm_mod(&mut a, &x, &mut want).unwrap();
            Simd.fm_mod(&mut b, &x, &mut got).unwrap();
            assert_eq!(want, got, "n={n} dev={dev}");
            assert_eq!(a.phase(), b.phase());
        }
    }
}

#[test]
fn fm_mod_extreme_inputs() {
    let mut x = vec![0.25f32; 100];
    x[5] = f32::NAN;
    x[17] = f32::INFINITY;
    x[18] = f32::NEG_INFINITY;
    x[40] = 3.0e38;
    x[41] = -3.0e38;
    x[60] = 1.0;
    x[61] = -1.0;
    for dev in [75_000.0, 1e12, 1e-9] {
        let (mut a, mut b) = (fm_state(dev, 240_000), fm_state(dev, 240_000));
        let (mut want, mut got) = (vec![C32::default(); 100], vec![C32::default(); 100]);
        Fallback.fm_mod(&mut a, &x, &mut want).unwrap();
        Simd.fm_mod(&mut b, &x, &mut got).unwrap();
        assert_eq!(bits(&want), bits(&got), "dev={dev}");
        assert_eq!(a.phase(), b.phase());
    }
    let q: Vec<Q15> = (0..100)
        .map(|i| Q15(if i % 2 == 0 { -32768 } else { 32767 }))
        .collect();
    let (mut a, mut b) = (fm_state(75_000.0, 240_000), fm_state(75_000.0, 240_000));
    let (mut want, mut got) = (vec![CQ15::default(); 100], vec![CQ15::default(); 100]);
    Fallback.fm_mod(&mut a, &q, &mut want).unwrap();
    Simd.fm_mod(&mut b, &q, &mut got).unwrap();
    assert_eq!(want, got);
}

#[test]
fn fm_mod_is_block_size_invariant() {
    let mut rng = Rng::new(15);
    let n = 3000;
    let x: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
    let start = fm_state(75_000.0, 240_000);
    let mut s1 = start;
    let mut whole = vec![C32::default(); n];
    Simd.fm_mod(&mut s1, &x, &mut whole).unwrap();
    let mut s2 = start;
    let mut parts = vec![C32::default(); n];
    let mut at = 0;
    for c in chunks(&mut rng, n) {
        Simd.fm_mod(&mut s2, &x[at..at + c], &mut parts[at..at + c])
            .unwrap();
        at += c;
    }
    assert_eq!(bits(&whole), bits(&parts));
    assert_eq!(s1.phase(), s2.phase());
}
