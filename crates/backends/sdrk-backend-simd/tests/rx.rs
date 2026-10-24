//! `mag_sq` and `quad_demod` against the fallback.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

mod common;

use common::{LENGTHS, Rng, c32_noise, chunks, cq15_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::demod::{QuadDemod, QuadDemodState};
use sdrk_core::kernels::spectrum::MagSq;
use sdrk_core::{C32, CQ15, Fallback, Q15};

#[test]
fn mag_sq_is_bit_exact() {
    let mut rng = Rng::new(30);
    for &n in &LENGTHS {
        let x = c32_noise(&mut rng, n, 3.0);
        let (mut want, mut got) = (vec![0f32; n], vec![0f32; n]);
        Fallback.mag_sq(&x, &mut want).unwrap();
        Simd.mag_sq(&x, &mut got).unwrap();
        // No multiply-add, so bit-exact also in the relaxed WASM build.
        let b = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(b(&want), b(&got), "n={n}");
        let q = cq15_noise(&mut rng, n);
        let (mut want, mut got) = (vec![Q15(0); n], vec![Q15(0); n]);
        Fallback.mag_sq(&q, &mut want).unwrap();
        Simd.mag_sq(&q, &mut got).unwrap();
        assert_eq!(want, got, "n={n}");
    }
}

#[test]
fn mag_sq_extremes() {
    let x = [
        C32::new(f32::NAN, 1.0),
        C32::new(f32::INFINITY, 0.0),
        C32::new(-0.0, -0.0),
        C32::new(1e30, 1e30),
        C32::new(1e-30, 1e-30),
    ];
    let mut want = [0f32; 5];
    let mut got = [0f32; 5];
    Fallback.mag_sq(&x, &mut want).unwrap();
    Simd.mag_sq(&x, &mut got).unwrap();
    for (w, g) in want.iter().zip(&got) {
        assert!(w.to_bits() == g.to_bits() || (w.is_nan() && g.is_nan()));
    }
    let q = [CQ15 {
        re: Q15(-32768),
        im: Q15(-32768),
    }; 20];
    let (mut want, mut got) = ([Q15(0); 20], [Q15(0); 20]);
    Fallback.mag_sq(&q, &mut want).unwrap();
    Simd.mag_sq(&q, &mut got).unwrap();
    assert_eq!(want, got);
    assert_eq!(got[0], Q15(32767));
}

fn fm_signal(rng: &mut Rng, n: usize) -> Vec<C32> {
    let mut phase = 0.0f64;
    (0..n)
        .map(|i| {
            phase += 0.3 * (i as f64 * 0.01).sin() + 0.02 * f64::from(rng.unit());
            let amp = 0.5 + 0.4 * f64::from(rng.unit());
            C32::new((amp * phase.cos()) as f32, (amp * phase.sin()) as f32)
        })
        .collect()
}

fn snr_db(want: &[f32], got: &[f32]) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    for (w, g) in want.iter().zip(got) {
        sig += f64::from(*w).powi(2);
        err += (f64::from(*w) - f64::from(*g)).powi(2);
    }
    if err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (sig / err).log10()
    }
}

#[test]
fn quad_demod_c32_matches_the_fallback() {
    let mut rng = Rng::new(31);
    for &n in &LENGTHS {
        let x = fm_signal(&mut rng, n);
        let mut a = QuadDemodState::<C32>::new();
        let mut b = a;
        let (mut want, mut got) = (vec![0f32; n], vec![0f32; n]);
        Fallback.quad_demod(&mut a, &x, &mut want, 2.5).unwrap();
        Simd.quad_demod(&mut b, &x, &mut got, 2.5).unwrap();
        if n > 0 {
            let snr = snr_db(&want, &got);
            assert!(snr > 120.0, "n={n} snr={snr}");
        }
        assert_eq!(a, b);
    }
}

#[test]
fn quad_demod_c32_special_values() {
    // Both products zero, one zero, negative zero, and the four quadrants.
    let x = [
        C32::new(0.0, 0.0),
        C32::new(1.0, 0.0),
        C32::new(0.0, 1.0),
        C32::new(-1.0, 0.0),
        C32::new(0.0, -1.0),
        C32::new(-1.0, -0.0),
        C32::new(-1.0, 0.0),
        C32::new(1e-20, 1e-20),
        C32::new(3.0, 4.0),
        C32::new(-3.0, 4.0),
        C32::new(3.0, -4.0),
        C32::new(-3.0, -4.0),
        C32::new(0.0, 0.0),
        C32::new(0.0, 0.0),
        C32::new(1.0, 1.0),
        C32::new(1.0, 1.0),
    ];
    let (mut a, mut b) = (QuadDemodState::<C32>::new(), QuadDemodState::<C32>::new());
    let (mut want, mut got) = (vec![0f32; x.len()], vec![0f32; x.len()]);
    Fallback.quad_demod(&mut a, &x, &mut want, 1.0).unwrap();
    Simd.quad_demod(&mut b, &x, &mut got, 1.0).unwrap();
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        assert!((w - g).abs() < 2e-6, "i={i} want={w} got={g}");
    }
}

/// The SNR of `quad_demod` against the f64 discriminator, without the first output (its
/// predecessor is the state).
fn quad_demod_snr(demod: &dyn QuadDemod<C32>, x: &[C32]) -> f64 {
    let mut state = QuadDemodState::<C32>::new();
    let mut got = vec![0f32; x.len()];
    demod.quad_demod(&mut state, x, &mut got, 1.0).unwrap();
    let want: Vec<f64> = x
        .windows(2)
        .map(|w| {
            let (a, b) = (w[1], w[0]);
            let re = f64::from(a.re) * f64::from(b.re) + f64::from(a.im) * f64::from(b.im);
            let im = f64::from(a.im) * f64::from(b.re) - f64::from(a.re) * f64::from(b.im);
            im.atan2(re)
        })
        .collect();
    let (mut sig, mut err) = (0f64, 0f64);
    for (w, g) in want.iter().zip(&got[1..]) {
        sig += w * w;
        err += (w - f64::from(*g)).powi(2);
    }
    10.0 * (sig / err).log10()
}

#[test]
fn quad_demod_c32_is_within_1_db_of_the_fallback_against_the_f64_reference() {
    // The fallback rounds `libm::atan2f` of the same `f32` products: the accuracy of the
    // libm-quality references. Small arguments (an FM carrier, as in the benchmark) and
    // arguments over the whole circle (noise).
    let mut rng = Rng::new(34);
    for case in 0..4 {
        let n = 20_000 + case;
        let x: Vec<C32> = if case % 2 == 0 {
            let mut phase = 0.0f64;
            (0..n)
                .map(|_| {
                    phase += 0.6 * f64::from(rng.unit());
                    C32::new((0.8 * phase.cos()) as f32, (0.8 * phase.sin()) as f32)
                })
                .collect()
        } else {
            c32_noise(&mut rng, n, 2.0)
        };
        let ours = quad_demod_snr(&Simd, &x);
        let fallback = quad_demod_snr(&Fallback, &x);
        assert!(
            ours > fallback - 1.0,
            "case {case}: {ours:.3} dB vs fallback {fallback:.3} dB"
        );
    }
}

#[test]
fn quad_demod_c32_axes_and_signed_zeros_match_the_fallback() {
    // Every pair of samples from the axes, signed zeros included, so that the products are
    // the signed zeros (both zero gives 0) and the exact multiples of pi/2. Each pattern runs
    // at every position of a vector, through the first vector, the main loop and the tail.
    let values = [0.0f32, -0.0, 1.0, -1.0, 0.5, -2.0];
    let axes: Vec<C32> = values
        .iter()
        .flat_map(|&re| values.iter().map(move |&im| C32::new(re, im)))
        .filter(|c| c.re == 0.0 || c.im == 0.0)
        .collect();
    let mut x = Vec::new();
    for a in &axes {
        for b in &axes {
            x.extend([*a, *b]);
        }
    }
    for shift in 0..17 {
        let x = &x[shift..];
        let (mut a, mut b) = (QuadDemodState::<C32>::new(), QuadDemodState::<C32>::new());
        let (mut want, mut got) = (vec![0f32; x.len()], vec![0f32; x.len()]);
        Fallback.quad_demod(&mut a, x, &mut want, 1.0).unwrap();
        Simd.quad_demod(&mut b, x, &mut got, 1.0).unwrap();
        for (i, (w, g)) in want.iter().zip(&got).enumerate() {
            assert!(
                w == g,
                "shift {shift} i={i}: {:?} {:?} want={w} got={g}",
                x.get(i.wrapping_sub(1)),
                x[i]
            );
        }
    }
}

#[test]
fn quad_demod_c32_is_block_size_invariant() {
    let mut rng = Rng::new(32);
    let n = 3000;
    let x = fm_signal(&mut rng, n);
    let mut a = QuadDemodState::<C32>::new();
    let mut b = a;
    let mut whole = vec![0f32; n];
    Simd.quad_demod(&mut a, &x, &mut whole, 1.0).unwrap();
    let mut parts = vec![0f32; n];
    let mut at = 0;
    for c in chunks(&mut rng, n) {
        Simd.quad_demod(&mut b, &x[at..at + c], &mut parts[at..at + c], 1.0)
            .unwrap();
        at += c;
    }
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&whole), bits(&parts));
    assert_eq!(a, b);
}

#[test]
fn quad_demod_cq15_is_within_one_lsb_of_the_fallback() {
    let mut rng = Rng::new(33);
    for &n in &LENGTHS {
        for gain in [32767i16, 16384, 1000, -20000] {
            let x = if n % 2 == 0 {
                cq15_noise(&mut rng, n)
            } else {
                // A slowly rotating phasor.
                (0..n)
                    .map(|i| {
                        let p = i as f64 * 0.4;
                        CQ15 {
                            re: Q15((20000.0 * p.cos()) as i16),
                            im: Q15((20000.0 * p.sin()) as i16),
                        }
                    })
                    .collect()
            };
            let mut a = QuadDemodState::<CQ15>::new();
            let mut b = a;
            let (mut want, mut got) = (vec![Q15(0); n], vec![Q15(0); n]);
            Fallback
                .quad_demod(&mut a, &x, &mut want, Q15(gain))
                .unwrap();
            Simd.quad_demod(&mut b, &x, &mut got, Q15(gain)).unwrap();
            for (i, (w, g)) in want.iter().zip(&got).enumerate() {
                let d = (i32::from(w.0) - i32::from(g.0)).abs();
                assert!(d <= 2, "n={n} gain={gain} i={i} want={} got={}", w.0, g.0);
            }
            assert_eq!(a, b);
        }
    }
}
