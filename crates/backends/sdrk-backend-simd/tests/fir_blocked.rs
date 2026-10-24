//! The register-blocked `C32` paths of `fir` and `fir_decimate` (transposed form for
//! consecutive outputs, chunk form for decimation) at the current level: every tap count
//! remainder of the vector widths, every small decimation factor, windows that reach into the
//! history or past the end of the input, block-size invariance and non-finite samples next to a
//! window; and the block-size invariance of the polyphase path (`resample`, `fir_interpolate`),
//! whose partial chunks are read in place with a mask where they can be, and that a non-finite
//! sample next to one of its windows does not leak into it.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

mod common;

use common::{Rng, bits, c32_noise, chunks, snr_db, taps_f32};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate, Resample, ResampleState};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::interp::{FirInterpState, FirInterpolate};
use sdrk_core::{C32, Fallback};

/// Tap counts with every remainder modulo 2, 4, 8 and 16 complex lanes, around the tap block of
/// 128 and beyond it.
fn tap_counts() -> Vec<usize> {
    let mut n: Vec<usize> = (1..=20).collect();
    n.extend([
        31, 32, 33, 63, 64, 65, 127, 128, 129, 130, 135, 255, 256, 257, 300,
    ]);
    n
}

/// `fir_decimate` of `x` in one call and in random blocks; the outputs and the final state.
fn decimate_both(
    rng: &mut Rng,
    taps: &[f32],
    factor: usize,
    x: &[C32],
) -> (Vec<C32>, Vec<C32>, Vec<C32>) {
    let n = taps.len();
    let mut a = FirDecimState::<C32, _, _>::new(taps.to_vec(), vec![C32::default(); n - 1], factor)
        .unwrap();
    let mut b = a.clone();
    let mut c = a.clone();
    let mut whole = vec![C32::default(); x.len()];
    let p = Simd.fir_decimate(&mut a, x, &mut whole).unwrap();
    assert_eq!(p.consumed, x.len());
    whole.truncate(p.produced);
    let mut parts = Vec::new();
    let mut at = 0;
    for len in chunks(rng, x.len()) {
        let mut out = vec![C32::default(); len];
        let q = Simd
            .fir_decimate(&mut b, &x[at..at + len], &mut out)
            .unwrap();
        assert_eq!(q.consumed, len);
        parts.extend_from_slice(&out[..q.produced]);
        at += len;
    }
    let mut want = vec![C32::default(); x.len()];
    let r = Fallback.fir_decimate(&mut c, x, &mut want).unwrap();
    assert_eq!(r, p);
    assert_eq!(a.countdown(), c.countdown());
    assert_eq!(b.countdown(), c.countdown());
    assert_eq!(bits(a.history()), bits(c.history()));
    assert_eq!(bits(b.history()), bits(c.history()));
    want.truncate(r.produced);
    (whole, parts, want)
}

#[test]
fn fir_matches_the_fallback_and_is_block_size_invariant_for_every_tap_count() {
    let mut rng = Rng::new(40);
    for n in tap_counts() {
        let taps = taps_f32(&mut rng, n);
        let len = 2 * n + 150;
        let x = c32_noise(&mut rng, len, 1.0);
        let mut a = FirState::<C32, _, _>::new(taps.clone(), vec![C32::default(); n - 1]).unwrap();
        let mut b = a.clone();
        let mut f = a.clone();
        let mut whole = vec![C32::default(); len];
        Simd.fir(&mut a, &x, &mut whole).unwrap();
        let mut parts = vec![C32::default(); len];
        let mut at = 0;
        for c in chunks(&mut rng, len) {
            Simd.fir(&mut b, &x[at..at + c], &mut parts[at..at + c])
                .unwrap();
            at += c;
        }
        let mut want = vec![C32::default(); len];
        Fallback.fir(&mut f, &x, &mut want).unwrap();
        assert_eq!(bits(&whole), bits(&parts), "taps={n}");
        let snr = snr_db(&want, &whole);
        assert!(snr > 125.0, "taps={n} snr={snr}");
        assert_eq!(a.history(), f.history());
        assert_eq!(b.history(), f.history());
    }
}

#[test]
fn fir_decimate_matches_the_fallback_and_is_block_size_invariant_for_every_factor() {
    let mut rng = Rng::new(41);
    for n in tap_counts() {
        let taps = taps_f32(&mut rng, n);
        for factor in 1..=16 {
            let len = 2 * n + 40 * factor + 50;
            let x = c32_noise(&mut rng, len, 1.0);
            let (whole, parts, want) = decimate_both(&mut rng, &taps, factor, &x);
            assert_eq!(bits(&whole), bits(&parts), "taps={n} m={factor}");
            let snr = snr_db(&want, &whole);
            assert!(snr > 125.0, "taps={n} m={factor} snr={snr}");
        }
    }
}

#[test]
fn fir_decimate_by_one_equals_fir_bit_for_bit() {
    let mut rng = Rng::new(42);
    for n in [1usize, 5, 64, 127, 129, 300] {
        let taps = taps_f32(&mut rng, n);
        let x = c32_noise(&mut rng, 700, 1.0);
        let mut f = FirState::<C32, _, _>::new(taps.clone(), vec![C32::default(); n - 1]).unwrap();
        let mut y = vec![C32::default(); x.len()];
        Simd.fir(&mut f, &x, &mut y).unwrap();
        let (whole, parts, _) = decimate_both(&mut rng, &taps, 1, &x);
        assert_eq!(bits(&whole), bits(&y), "taps={n}");
        assert_eq!(bits(&parts), bits(&y), "taps={n}");
    }
}

/// Outputs whose window does not hold the non-finite sample stay finite and equal to the
/// outputs of the same input with a zero there; the others are not finite.
fn check_isolated(n: usize, factor: usize, at: usize, bad: f32, rng: &mut Rng) {
    let taps: Vec<f32> = (0..n).map(|i| 0.5 + (i % 7) as f32 / 16.0).collect();
    let len = 3 * n + 64;
    let mut x = c32_noise(rng, len, 1.0);
    x[at] = C32::new(0.0, 0.0);
    let (clean, _, _) = decimate_both(rng, &taps, factor, &x);
    x[at] = C32::new(bad, bad);
    let (whole, parts, _) = decimate_both(rng, &taps, factor, &x);
    assert_eq!(bits(&whole), bits(&parts), "taps={n} m={factor} at={at}");
    for (j, (y, c)) in whole.iter().zip(&clean).enumerate() {
        // Output `j` is input index `j * factor` and reads the inputs `j m - n + 1 ..= j m`.
        let newest = j * factor;
        let inside = at <= newest && at + n > newest;
        let finite = y.re.is_finite() && y.im.is_finite();
        assert_eq!(finite, !inside, "taps={n} m={factor} at={at} j={j}");
        if !inside {
            assert_eq!(
                bits(&[*y]),
                bits(&[*c]),
                "taps={n} m={factor} at={at} j={j}"
            );
        }
    }
}

#[test]
fn a_non_finite_sample_next_to_a_window_does_not_leak_into_it() {
    let mut rng = Rng::new(43);
    for n in [1usize, 3, 7, 9, 15, 17, 64, 127, 129] {
        for factor in [1usize, 2, 3, 10] {
            for bad in [f32::NAN, f32::INFINITY] {
                // Next to the windows of the first outputs (in the history part) and of later
                // ones (inside the input), and near the end.
                for at in [0, 1, n - 1, n, n + 1, 2 * n + 5, 3 * n + 62, 3 * n + 63] {
                    check_isolated(n, factor, at, bad, &mut rng);
                }
            }
        }
    }
}

#[test]
fn resample_is_block_size_invariant() {
    let mut rng = Rng::new(44);
    for &(up, down) in &[(1usize, 1usize), (2, 1), (3, 2), (5, 7), (10, 3)] {
        for n in [1usize, 5, 16, 31, 64, 129, 300] {
            let taps = taps_f32(&mut rng, n);
            let x = c32_noise(&mut rng, 600, 1.0);
            let mut a = ResampleState::<C32, _, _>::new(
                taps.clone(),
                vec![C32::default(); n.div_ceil(up)],
                up,
                down,
            )
            .unwrap();
            let mut b = a.clone();
            let room = x.len() * up / down + 8;
            let mut whole = vec![C32::default(); room];
            let p = Simd.resample(&mut a, &x, &mut whole).unwrap();
            assert_eq!(p.consumed, x.len());
            whole.truncate(p.produced);
            let mut parts = Vec::new();
            let mut at = 0;
            for len in chunks(&mut rng, x.len()) {
                let mut out = vec![C32::default(); len * up / down + 8];
                let q = Simd.resample(&mut b, &x[at..at + len], &mut out).unwrap();
                assert_eq!(q.consumed, len);
                parts.extend_from_slice(&out[..q.produced]);
                at += len;
            }
            assert_eq!(bits(&whole), bits(&parts), "L={up} M={down} taps={n}");
        }
    }
}

/// `resample` of `x` by `up / down` with the backend `k`, in one call.
fn resample_with<K: Resample<C32>>(
    k: &K,
    taps: &[f32],
    up: usize,
    down: usize,
    x: &[C32],
) -> Vec<C32> {
    let n = taps.len();
    let mut s = ResampleState::<C32, _, _>::new(
        taps.to_vec(),
        vec![C32::default(); n.div_ceil(up)],
        up,
        down,
    )
    .unwrap();
    let mut out = vec![C32::default(); x.len() * up / down + 8];
    let p = k.resample(&mut s, x, &mut out).unwrap();
    assert_eq!(p.consumed, x.len());
    out.truncate(p.produced);
    out
}

#[test]
fn a_non_finite_sample_next_to_a_polyphase_window_does_not_leak_into_it() {
    let mut rng = Rng::new(45);
    for &(up, down) in &[(1usize, 1usize), (2, 1), (3, 2), (5, 7), (4, 3)] {
        for n in [1usize, 3, 5, 7, 9, 17, 31, 33, 64] {
            let taps: Vec<f32> = (0..n).map(|i| 0.5 + (i % 7) as f32 / 16.0).collect();
            let len = 2 * n + 40;
            for bad in [f32::NAN, f32::INFINITY] {
                for at in [0, 1, n / 2, n, n + 1, len - 2, len - 1] {
                    let mut x = c32_noise(&mut rng, len, 1.0);
                    x[at] = C32::new(0.0, 0.0);
                    let clean = resample_with(&Simd, &taps, up, down, &x);
                    x[at] = C32::new(bad, bad);
                    let got = resample_with(&Simd, &taps, up, down, &x);
                    let want = resample_with(&Fallback, &taps, up, down, &x);
                    assert_eq!(got.len(), want.len());
                    for (j, ((y, w), c)) in got.iter().zip(&want).zip(&clean).enumerate() {
                        // The fallback reads exactly the window, so it shows which outputs hold
                        // the non-finite sample.
                        let inside = !(w.re.is_finite() && w.im.is_finite());
                        let finite = y.re.is_finite() && y.im.is_finite();
                        let at_ = format!("L={up} M={down} taps={n} at={at} j={j}");
                        assert_eq!(finite, !inside, "{at_}");
                        if !inside {
                            assert_eq!(bits(&[*y]), bits(&[*c]), "{at_}");
                        }
                    }
                }
            }
        }
    }
}

/// `fir_interpolate` of `x` by `up` with the backend `k`, in one call.
fn interpolate_with<K: FirInterpolate<C32>>(k: &K, taps: &[f32], up: usize, x: &[C32]) -> Vec<C32> {
    let n = taps.len();
    let mut s =
        FirInterpState::<C32, _, _>::new(taps.to_vec(), vec![C32::default(); n.div_ceil(up)], up)
            .unwrap();
    let mut out = vec![C32::default(); x.len() * up];
    let p = k.fir_interpolate(&mut s, x, &mut out).unwrap();
    assert_eq!(p.consumed, x.len());
    out.truncate(p.produced);
    out
}

#[test]
fn fir_interpolate_keeps_a_non_finite_sample_inside_its_windows() {
    let mut rng = Rng::new(46);
    for up in [2usize, 3, 8] {
        for n in [4usize, 9, 24, 33] {
            let taps: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 / 16.0).collect();
            let len = n + 30;
            let mut x = c32_noise(&mut rng, len, 1.0);
            x[len / 2] = C32::new(f32::NAN, f32::NAN);
            let got = interpolate_with(&Simd, &taps, up, &x);
            let want = interpolate_with(&Fallback, &taps, up, &x);
            assert_eq!(got.len(), want.len());
            for (j, (y, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(y.re.is_finite(), w.re.is_finite(), "L={up} taps={n} j={j}");
            }
        }
    }
}
