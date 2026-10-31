//! Tests of the halfband references of M23 (liquid-dsp's `resamp2` and `firhilbf`, libairspy's
//! converters) against the definition in `f64`: the output of every reference is `h` convolved
//! with the input and kept at the even positions (for the Hilbert transform and the converters
//! after the shift by fs / 4), up to the fixed mapping that the benchmark applies (`firhilbf`:
//! `j / centre` times the definition, libairspy: `-conj`). Run through the benchmark workspace
//! with `SDRK_REFS_DIR` set as xtask sets it; a library that is not linked skips its tests.

use cref_sys::{AirspyFloat, AirspyInt16, LiquidFirhilbf, LiquidResamp2Crcf, LiquidResamp2Rrrf};
use std::f64::consts::PI;

const N: usize = 47;

/// A 47-tap halfband filter: a Hamming-windowed sinc with the zeros at the even offsets,
/// scaled to the sum 1 (centre tap about 0.5).
fn taps() -> Vec<f64> {
    let mut h = vec![0.0f64; N];
    for (i, t) in h.iter_mut().enumerate() {
        let k = i as f64 - 23.0;
        let window = 0.54 + 0.46 * (PI * k / 24.0).cos();
        *t = if i == 23 {
            0.5
        } else if i.abs_diff(23) % 2 == 0 {
            0.0
        } else {
            (PI * k / 2.0).sin() / (PI * k) * window
        };
    }
    let sum: f64 = h.iter().sum();
    h.iter().map(|v| v / sum).collect()
}

fn signal(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (f64::from(x) / f64::from(u32::MAX) - 0.5) as f32
        })
        .collect()
}

/// The prototype of `4 m + 1 = 49` taps: the 47 taps between two zeros, divided by the centre.
fn prototype(h: &[f64]) -> Vec<f32> {
    let mut p = vec![0.0f32; 49];
    for (d, s) in p[1..48].iter_mut().zip(h) {
        *d = (s / h[23]) as f32;
    }
    p
}

/// `y[j] = sum_k h[k] u[2 j - k]` with zero history.
fn decimate(h: &[f64], u: &[[f64; 2]]) -> Vec<[f64; 2]> {
    (0..u.len() / 2)
        .map(|j| {
            let mut acc = [0.0f64; 2];
            for (k, t) in h.iter().enumerate() {
                if let Some(v) = (2 * j).checked_sub(k).and_then(|n| u.get(n)) {
                    acc[0] += t * v[0];
                    acc[1] += t * v[1];
                }
            }
            acc
        })
        .collect()
}

/// `x[n] (-j)^n`.
fn shifted(x: &[f32]) -> Vec<[f64; 2]> {
    x.iter()
        .enumerate()
        .map(|(n, v)| {
            let v = f64::from(*v);
            match n % 4 {
                0 => [v, 0.0],
                1 => [0.0, -v],
                2 => [-v, 0.0],
                _ => [0.0, v],
            }
        })
        .collect()
}

fn max_error(want: &[[f64; 2]], got: &[[f64; 2]]) -> f64 {
    want.iter()
        .zip(got)
        .map(|(w, g)| (w[0] - g[0]).abs().max((w[1] - g[1]).abs()))
        .fold(0.0, f64::max)
}

#[test]
fn liquid_resamp2_matches_the_definition_with_both_dot_products() {
    let h = taps();
    let x = signal(512, 1);
    let xc: Vec<[f32; 2]> = x.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
    let u: Vec<[f64; 2]> = xc
        .iter()
        .map(|p| [f64::from(p[0]), f64::from(p[1])])
        .collect();
    let want = decimate(&h, &u);
    let real: Vec<[f64; 2]> = x.iter().map(|v| [f64::from(*v), 0.0]).collect();
    let want_real = decimate(&h, &real);
    for simd in [false, true] {
        let Some(mut q) = LiquidResamp2Crcf::new(&prototype(&h), h[23] as f32, simd) else {
            eprintln!("skipped: liquid-dsp's halfband decimators are not linked");
            return;
        };
        let mut input = xc.clone();
        let mut y = vec![[0.0f32; 2]; 128];
        q.decim(&mut input, &mut y);
        let got: Vec<[f64; 2]> = y
            .iter()
            .map(|p| [f64::from(p[0]), f64::from(p[1])])
            .collect();
        assert!(max_error(&want, &got) < 1e-5, "crcf simd={simd}");
        // The second call continues the stream; reset starts over.
        q.reset();
        let mut again = vec![[0.0f32; 2]; 128];
        let mut input = xc.clone();
        q.decim(&mut input, &mut again);
        assert_eq!(y, again);

        let mut q = LiquidResamp2Rrrf::new(&prototype(&h), h[23] as f32, simd).expect("rrrf");
        let mut input = x.clone();
        let mut y = vec![0.0f32; 256];
        q.decim(&mut input, &mut y);
        let got: Vec<[f64; 2]> = y.iter().map(|v| [f64::from(*v), 0.0]).collect();
        assert!(max_error(&want_real, &got) < 1e-5, "rrrf simd={simd}");
    }
}

#[test]
fn liquid_firhilbf_is_j_over_the_centre_tap_times_real_to_complex() {
    let h = taps();
    let x = signal(512, 2);
    let want = decimate(&h, &shifted(&x));
    for simd in [false, true] {
        let Some(mut q) = LiquidFirhilbf::new(&prototype(&h), simd) else {
            eprintln!("skipped: liquid-dsp's halfband decimators are not linked");
            return;
        };
        let mut input = x.clone();
        let mut y = vec![[0.0f32; 2]; 256];
        q.decim(&mut input, &mut y);
        // The benchmark maps `-j c z`.
        let c = h[23];
        let got: Vec<[f64; 2]> = y
            .iter()
            .map(|p| [c * f64::from(p[1]), -c * f64::from(p[0])])
            .collect();
        assert!(max_error(&want, &got) < 1e-5, "firhilbf simd={simd}");
    }
}

#[test]
fn libairspy_converters_are_minus_conj_of_real_to_complex() {
    let h = taps();
    let x = signal(512, 3);
    let want = decimate(&h, &shifted(&x));
    let taps32: Vec<f32> = h.iter().map(|v| *v as f32).collect();
    for sse2 in [false, true] {
        let Some(mut q) = AirspyFloat::new(&taps32, sse2) else {
            eprintln!("skipped: libairspy's converter (sse2 = {sse2}) is not available");
            continue;
        };
        let mut y = vec![0.0f32; 512];
        q.convert(&x, &mut y);
        let got: Vec<[f64; 2]> = y
            .chunks_exact(2)
            .map(|p| [-f64::from(p[0]), f64::from(p[1])])
            .collect();
        assert!(max_error(&want, &got) < 1e-5, "float sse2={sse2}");
    }
    let taps16: Vec<i16> = h.iter().map(|v| (v * 32768.0).round() as i16).collect();
    let Some(mut q) = AirspyInt16::new(&taps16) else {
        eprintln!("skipped: libairspy's converters are not linked");
        return;
    };
    let x16: Vec<i16> = x.iter().map(|v| (f64::from(*v) * 32768.0) as i16).collect();
    let mut y = vec![0i16; 512];
    q.convert(&x16, &mut y);
    // Truncation and the fixed centre tap: a lower accuracy class, still better than 1e-3.
    let scale = 1.0 / 32768.0;
    let got: Vec<[f64; 2]> = y
        .chunks_exact(2)
        .map(|p| [-f64::from(p[0]) * scale, f64::from(p[1]) * scale])
        .collect();
    assert!(max_error(&want, &got) < 1e-3, "int16");
    assert!(AirspyInt16::new(&taps16[..46]).is_none());
}
