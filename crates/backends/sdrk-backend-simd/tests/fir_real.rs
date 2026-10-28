//! `resample_real` and `fir_interpolate_real` against the fallback.
//!
//! `Q15` is compared bit for bit; `f32` by SNR against the fallback (the sums are formed in a
//! different order) and, for the block-size invariance and the non-finite samples, bit for bit
//! between runs of the backend itself.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]

mod common;

use common::{LENGTHS, Rng, chunks, db, taps_f32};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::decim::{ResampleReal, ResampleRealState};
use sdrk_core::kernels::interp::{FirInterpRealState, FirInterpolateReal};
use sdrk_core::{Fallback, Progress, Q15};

/// The ratios: interpolators (`M = 1`, the transposed form), decimators and rational ratios
/// (the chunk form), and one with more phases than taps.
const RATIOS: [(usize, usize); 10] = [
    (1, 1),
    (2, 1),
    (3, 1),
    (10, 1),
    (1, 5),
    (3, 2),
    (5, 7),
    (4, 3),
    (147, 160),
    (160, 147),
];

/// Tap counts: shorter than a vector, every chunk remainder, several tap blocks per branch.
const TAP_COUNTS: [usize; 10] = [1, 3, 4, 9, 17, 31, 64, 200, 600, 1100];

fn f32_noise(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.unit()).collect()
}

fn q15_noise(rng: &mut Rng, n: usize, extreme: bool) -> Vec<Q15> {
    (0..n)
        .map(|i| match (extreme, i % 4) {
            (true, 0) => Q15(-32768),
            (true, 1) => Q15(32767),
            _ => Q15(rng.next_u32() as i16),
        })
        .collect()
}

fn taps_q15(rng: &mut Rng, n: usize, extreme: bool) -> Vec<Q15> {
    (0..n)
        .map(|i| {
            if extreme && i % 3 == 0 {
                Q15(-32768)
            } else {
                Q15((rng.next_u32() as i16) / 4)
            }
        })
        .collect()
}

fn snr(want: &[f32], got: &[f32]) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    for (w, g) in want.iter().zip(got) {
        sig += f64::from(*w).powi(2);
        err += (f64::from(*w) - f64::from(*g)).powi(2);
    }
    db(sig, err)
}

fn state<T: sdrk_core::RealSample + Default>(
    taps: &[T],
    up: usize,
    down: usize,
) -> ResampleRealState<T, Vec<T>, Vec<T>> {
    let hist = vec![T::default(); taps.len().div_ceil(up)];
    ResampleRealState::new(taps.to_vec(), hist, up, down).unwrap()
}

/// The output room of a call: none, a few samples, or enough for all.
fn room(len: usize, up: usize, down: usize, which: usize) -> usize {
    match which {
        0 => 0,
        1 => 5,
        _ => len * up / down + 4,
    }
}

#[test]
fn resample_real_f32_matches_the_fallback() {
    let mut rng = Rng::new(61);
    for &(up, down) in &RATIOS {
        for &n in &TAP_COUNTS {
            for which in 0..3 {
                let len = 150;
                let taps = taps_f32(&mut rng, n);
                let x = f32_noise(&mut rng, len);
                let (mut a, mut b) = (state(&taps, up, down), state(&taps, up, down));
                let out_len = room(len, up, down, which);
                let (mut want, mut got) = (vec![0f32; out_len], vec![0f32; out_len]);
                let pw = Fallback.resample_real(&mut a, &x, &mut want).unwrap();
                let pg = Simd.resample_real(&mut b, &x, &mut got).unwrap();
                let at = format!("L={up} M={down} taps={n} room={out_len}");
                assert_eq!(pw, pg, "{at}");
                assert_eq!(a.phase(), b.phase(), "{at}");
                assert_eq!(a.history(), b.history(), "{at}");
                // Nothing beyond the outputs produced is written.
                assert!(got[pg.produced..].iter().all(|&v| v == 0.0), "{at}");
                if pw.produced > 0 {
                    let s = snr(&want[..pw.produced], &got[..pw.produced]);
                    assert!(s > 120.0, "{at} snr={s}");
                }
            }
        }
    }
}

#[test]
fn resample_real_f32_covers_every_input_length() {
    let mut rng = Rng::new(62);
    for &(up, down) in &[(1usize, 1usize), (2, 1), (3, 2), (1, 5)] {
        for n in [5usize, 31, 64] {
            for len in LENGTHS {
                let taps = taps_f32(&mut rng, n);
                let x = f32_noise(&mut rng, len);
                let (mut a, mut b) = (state(&taps, up, down), state(&taps, up, down));
                let out_len = len * up / down + 4;
                let (mut want, mut got) = (vec![0f32; out_len], vec![0f32; out_len]);
                let pw = Fallback.resample_real(&mut a, &x, &mut want).unwrap();
                let pg = Simd.resample_real(&mut b, &x, &mut got).unwrap();
                assert_eq!(pw, pg, "L={up} M={down} taps={n} len={len}");
                if pw.produced > 0 {
                    let s = snr(&want[..pw.produced], &got[..pw.produced]);
                    assert!(s > 120.0, "L={up} M={down} taps={n} len={len} snr={s}");
                }
            }
        }
    }
}

/// `resample_real` of `x` with the backend `k` in the given chunks; the outputs and the
/// progress of every call.
fn run_chunks<T: sdrk_core::RealSample + Default, K: ResampleReal<T>>(
    k: &K,
    s: &mut ResampleRealState<T, Vec<T>, Vec<T>>,
    x: &[T],
    sizes: &[usize],
) -> Vec<T> {
    let (up, down) = (s.up(), s.down());
    let mut out = Vec::new();
    let mut at = 0;
    for &len in sizes {
        let mut y = vec![T::default(); len * up / down + 8];
        let p: Progress = k.resample_real(s, &x[at..at + len], &mut y).unwrap();
        assert_eq!(p.consumed, len);
        out.extend_from_slice(&y[..p.produced]);
        at += len;
    }
    out
}

#[test]
fn resample_real_f32_is_block_size_invariant() {
    let mut rng = Rng::new(63);
    for &(up, down) in &RATIOS {
        for n in [1usize, 5, 16, 31, 64, 129, 300] {
            let taps = taps_f32(&mut rng, n);
            let x = f32_noise(&mut rng, 600);
            let whole = run_chunks(&Simd, &mut state(&taps, up, down), &x, &[x.len()]);
            let sizes = chunks(&mut rng, x.len());
            let parts = run_chunks(&Simd, &mut state(&taps, up, down), &x, &sizes);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&whole), bits(&parts), "L={up} M={down} taps={n}");
        }
    }
}

#[test]
fn a_non_finite_sample_stays_inside_its_real_windows() {
    let mut rng = Rng::new(64);
    for &(up, down) in &[(1usize, 1usize), (2, 1), (3, 2), (5, 7), (4, 3), (1, 3)] {
        for n in [1usize, 3, 5, 7, 9, 17, 31, 33, 64] {
            let taps: Vec<f32> = (0..n).map(|i| 0.5 + (i % 7) as f32 / 16.0).collect();
            let len = 2 * n + 40;
            for bad in [f32::NAN, f32::INFINITY] {
                for at in [0, 1, n / 2, n, n + 1, len - 2, len - 1] {
                    let mut x = f32_noise(&mut rng, len);
                    x[at] = 0.0;
                    let all = [len];
                    let clean = run_chunks(&Simd, &mut state(&taps, up, down), &x, &all);
                    x[at] = bad;
                    let got = run_chunks(&Simd, &mut state(&taps, up, down), &x, &all);
                    let want = run_chunks(&Fallback, &mut state(&taps, up, down), &x, &all);
                    assert_eq!(got.len(), want.len());
                    for (j, ((y, w), c)) in got.iter().zip(&want).zip(&clean).enumerate() {
                        // The fallback reads exactly the window, so it shows which outputs hold
                        // the non-finite sample.
                        let inside = !w.is_finite();
                        let place = format!("L={up} M={down} taps={n} at={at} j={j}");
                        assert_eq!(y.is_finite(), !inside, "{place}");
                        if !inside {
                            assert_eq!(y.to_bits(), c.to_bits(), "{place}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn resample_real_q15_is_bit_exact() {
    let mut rng = Rng::new(65);
    for &(up, down) in &RATIOS {
        // 1100 taps at L = 1 do not fit the prepared storage and run on the fallback.
        for &n in &TAP_COUNTS {
            for extreme in [false, true] {
                for which in 0..3 {
                    let len = 150;
                    let taps = taps_q15(&mut rng, n, extreme);
                    let x = q15_noise(&mut rng, len, extreme);
                    let (mut a, mut b) = (state(&taps, up, down), state(&taps, up, down));
                    let out_len = room(len, up, down, which);
                    let (mut want, mut got) = (vec![Q15(0); out_len], vec![Q15(0); out_len]);
                    let pw = Fallback.resample_real(&mut a, &x, &mut want).unwrap();
                    let pg = Simd.resample_real(&mut b, &x, &mut got).unwrap();
                    let at = format!("L={up} M={down} taps={n} room={out_len} {extreme}");
                    assert_eq!(pw, pg, "{at}");
                    assert_eq!(want, got, "{at}");
                    assert_eq!(a.phase(), b.phase(), "{at}");
                    assert_eq!(a.history(), b.history(), "{at}");
                }
            }
        }
    }
}

#[test]
fn resample_real_q15_is_bit_exact_for_every_input_length_and_split() {
    let mut rng = Rng::new(66);
    for &(up, down) in &[(1usize, 1usize), (2, 1), (3, 2), (1, 5), (5, 7)] {
        for n in [1usize, 7, 31, 64, 129] {
            for len in LENGTHS {
                let taps = taps_q15(&mut rng, n, false);
                let x = q15_noise(&mut rng, len, false);
                let want = run_chunks(&Fallback, &mut state(&taps, up, down), &x, &[len]);
                let got = run_chunks(&Simd, &mut state(&taps, up, down), &x, &[len]);
                assert_eq!(want, got, "L={up} M={down} taps={n} len={len}");
                let sizes = chunks(&mut rng, len);
                let parts = run_chunks(&Simd, &mut state(&taps, up, down), &x, &sizes);
                assert_eq!(want, parts, "L={up} M={down} taps={n} len={len} split");
            }
        }
    }
}

#[test]
fn fir_interpolate_real_matches_the_fallback() {
    let mut rng = Rng::new(67);
    for up in [1usize, 2, 4, 5, 10] {
        for n in [1usize, 4, 9, 40, 241, 600] {
            let len = 100;
            let hist = n.div_ceil(up);
            let taps = taps_f32(&mut rng, n);
            let x = f32_noise(&mut rng, len);
            let mut a =
                FirInterpRealState::<f32, _, _>::new(taps.clone(), vec![0.0; hist], up).unwrap();
            let mut b = a.clone();
            let (mut want, mut got) = (vec![0f32; len * up], vec![0f32; len * up]);
            let pw = Fallback
                .fir_interpolate_real(&mut a, &x, &mut want)
                .unwrap();
            let pg = Simd.fir_interpolate_real(&mut b, &x, &mut got).unwrap();
            assert_eq!(pw, pg, "L={up} taps={n}");
            let s = snr(&want, &got);
            assert!(s > 120.0, "L={up} taps={n} snr={s}");

            let taps = taps_q15(&mut rng, n, up % 2 == 0);
            let x = q15_noise(&mut rng, len, up % 2 == 1);
            let mut a =
                FirInterpRealState::<Q15, _, _>::new(taps.clone(), vec![Q15(0); hist], up).unwrap();
            let mut b = a.clone();
            let (mut want, mut got) = (vec![Q15(0); len * up], vec![Q15(0); len * up]);
            let pw = Fallback
                .fir_interpolate_real(&mut a, &x, &mut want)
                .unwrap();
            let pg = Simd.fir_interpolate_real(&mut b, &x, &mut got).unwrap();
            assert_eq!(pw, pg, "L={up} taps={n}");
            assert_eq!(want, got, "L={up} taps={n}");
        }
    }
}
