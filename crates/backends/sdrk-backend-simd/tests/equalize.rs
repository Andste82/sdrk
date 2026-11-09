//! `dot_complex` and `equalize` (M43) against the fallback, bit for bit at the level of the
//! running build: `dot_complex` at every length up to 200 (every tail of the blocks of eight
//! samples) and at every start offset within a vector, with mixed magnitudes and with special
//! values (signed zeros, infinities, NaN, denormals, huge values); `equalize` with every
//! constellation and tap count class, with and without feedback taps and phase tracker, on noisy
//! streams, with step sizes at both ends, and in chunks.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::float_cmp
)]

mod common;

use common::{Rng, c32_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::dot::DotComplex;
use sdrk_core::kernels::equalize::{EqConstellation, EqParams, Equalize, EqualizeState};
use sdrk_core::{C32, Error, Fallback};

/// Start offsets in samples: the kernels must not depend on the alignment of the buffers.
const OFFSETS: [usize; 4] = [0, 1, 2, 3];

/// The bits of a result, with every NaN folded to one value (the sign and payload of a NaN are
/// not part of the contract).
fn bits(v: C32) -> (u32, u32) {
    let b = |x: f32| if x.is_nan() { u32::MAX } else { x.to_bits() };
    (b(v.re), b(v.im))
}

/// Samples of mixed magnitudes, so that the order of the additions shows in the bits.
fn mixed(rng: &mut Rng, n: usize) -> Vec<C32> {
    let scales = [1.0f32, -1e-3, 1e4, -7.5, 3e-20, 1e19];
    (0..n)
        .map(|_| {
            let s = scales[rng.below(scales.len())];
            C32::new(rng.unit() * s, rng.unit() * s)
        })
        .collect()
}

/// A special value or a plain one.
fn special(rng: &mut Rng) -> f32 {
    let v = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1e-40,
        -1e-45,
        f32::MAX,
        f32::MIN,
        1.0,
    ];
    if rng.below(4) == 0 {
        v[rng.below(v.len())]
    } else {
        rng.unit()
    }
}

#[test]
fn dot_complex_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(596);
    for n in 0..=200 {
        let x = mixed(&mut rng, n + 3);
        let w = mixed(&mut rng, n + 3);
        for off in OFFSETS {
            let (xs, ws) = (&x[off..off + n], &w[3 - off..3 - off + n]);
            assert_eq!(
                bits(Simd.dot_complex(xs, ws).unwrap()),
                bits(Fallback.dot_complex(xs, ws).unwrap()),
                "n {n} offset {off}"
            );
        }
    }
}

#[test]
fn dot_complex_is_bit_exact_with_special_values() {
    let mut rng = Rng::new(597);
    for n in [0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 33, 63, 64, 65] {
        for _ in 0..200 {
            let x: Vec<C32> = (0..n)
                .map(|_| C32::new(special(&mut rng), special(&mut rng)))
                .collect();
            let w: Vec<C32> = (0..n)
                .map(|_| C32::new(special(&mut rng), special(&mut rng)))
                .collect();
            assert_eq!(
                bits(Simd.dot_complex(&x, &w).unwrap()),
                bits(Fallback.dot_complex(&x, &w).unwrap()),
                "n {n}"
            );
        }
    }
    // Signed zeros: the sum of `-0` products is `+0`, as in the fallback.
    let z = [C32::new(-0.0, -0.0); 3];
    let one = [C32::new(1.0, 0.0); 3];
    assert_eq!(
        bits(Simd.dot_complex(&z, &one).unwrap()),
        bits(Fallback.dot_complex(&z, &one).unwrap())
    );
    assert_eq!(Simd.dot_complex(&[], &[]).unwrap().re.to_bits(), 0);
}

#[test]
fn dot_complex_checks_the_lengths() {
    let x = [C32::new(1.0, 0.0); 4];
    assert_eq!(
        Simd.dot_complex(&x, &x[..3]),
        Err(Error::LengthMismatch {
            expected: 4,
            got: 3
        })
    );
}

/// A noisy QPSK, BPSK or 8PSK stream through a short channel, scaled by `gain`.
fn stream(rng: &mut Rng, n: usize, gain: f32) -> Vec<C32> {
    let noise = c32_noise(rng, n, 0.1);
    let mut prev = C32::new(0.0, 0.0);
    (0..n)
        .map(|k| {
            let a = rng.below(8) as f32 * core::f32::consts::FRAC_PI_4 + 0.3;
            let s = C32::new(libm::cosf(a), libm::sinf(a));
            let y = s + prev * C32::new(0.3, -0.2) + noise[k];
            prev = s;
            y * gain
        })
        .collect()
}

#[test]
fn equalize_is_bit_exact_with_the_fallback() {
    let mut rng = Rng::new(596);
    let constellations = [
        EqConstellation::Bpsk,
        EqConstellation::Qpsk,
        EqConstellation::Psk8,
        EqConstellation::Qam16,
        EqConstellation::Qam64,
    ];
    for (case, taps) in [3usize, 5, 7, 9, 11, 15, 17, 23, 31, 63]
        .into_iter()
        .enumerate()
    {
        let c = constellations[case % 5];
        let feedback = [0usize, 3, 0, 5, 1][case % 5].min(63 - taps);
        for (mu_cma, mu_dd, cma, phase_bw) in [
            (0.05, 0.02, 300u32, 0.02),
            (1.0, 1.0, 50, 0.2),
            (1e-5, 1e-5, 0, 0.0),
        ] {
            let gain = [1.0f32, 0.5, 3.0][case % 3];
            let input = stream(&mut rng, 1500, gain);
            let mut a = EqualizeState::new(
                vec![C32::default(); taps + feedback],
                vec![C32::default(); taps + feedback],
                EqParams {
                    constellation: c,
                    feedback_taps: feedback,
                    mu_cma,
                    mu_dd,
                    cma_symbols: cma,
                    phase_bw,
                },
            )
            .unwrap();
            let mut b = a.clone();
            let mut out_a = vec![C32::default(); input.len()];
            let mut out_b = vec![C32::default(); input.len()];
            Fallback.equalize(&mut a, &input, &mut out_a).unwrap();
            // The SIMD run in chunks of 1 to 97 symbols.
            let mut at = 0;
            while at < input.len() {
                let len = (1 + rng.below(97)).min(input.len() - at);
                Simd.equalize(&mut b, &input[at..at + len], &mut out_b[at..at + len])
                    .unwrap();
                at += len;
            }
            let ob: Vec<_> = out_b.iter().map(|&v| bits(v)).collect();
            let oa: Vec<_> = out_a.iter().map(|&v| bits(v)).collect();
            assert_eq!(oa, ob, "taps {taps} mu {mu_cma}");
            let wa: Vec<_> = a.weights().iter().map(|&v| bits(v)).collect();
            let wb: Vec<_> = b.weights().iter().map(|&v| bits(v)).collect();
            assert_eq!(wa, wb, "taps {taps}");
            assert_eq!((a.mse().to_bits(), a.mode()), (b.mse().to_bits(), b.mode()));
            assert_eq!(
                (a.phase(), a.frequency().to_bits()),
                (b.phase(), b.frequency().to_bits())
            );
            let ha: Vec<_> = a.history().iter().map(|&v| bits(v)).collect();
            let hb: Vec<_> = b.history().iter().map(|&v| bits(v)).collect();
            assert_eq!(ha, hb, "taps {taps}");
        }
    }
}

#[test]
fn equalize_is_bit_exact_with_special_inputs() {
    let mut rng = Rng::new(597);
    let input: Vec<C32> = (0..3000)
        .map(|_| C32::new(special(&mut rng), special(&mut rng)))
        .collect();
    for c in [
        EqConstellation::Qpsk,
        EqConstellation::Psk8,
        EqConstellation::Qam16,
        EqConstellation::Qam64,
    ] {
        let mut a = EqualizeState::new(
            [C32::default(); 13],
            [C32::default(); 13],
            EqParams {
                constellation: c,
                feedback_taps: 2,
                mu_cma: 0.1,
                mu_dd: 0.1,
                cma_symbols: 100,
                phase_bw: 0.05,
            },
        )
        .unwrap();
        let mut b = a.clone();
        let mut out_a = vec![C32::default(); input.len()];
        let mut out_b = vec![C32::default(); input.len()];
        Fallback.equalize(&mut a, &input, &mut out_a).unwrap();
        Simd.equalize(&mut b, &input, &mut out_b).unwrap();
        let oa: Vec<_> = out_a.iter().map(|&v| bits(v)).collect();
        let ob: Vec<_> = out_b.iter().map(|&v| bits(v)).collect();
        assert_eq!(oa, ob);
        assert!(
            a.weights()
                .iter()
                .all(|w| w.re.is_finite() && w.im.is_finite())
        );
    }
}
