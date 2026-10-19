//! `fir`, `fir_decimate`, `resample` and `fir_interpolate` against the fallback.
//!
//! `CQ15` is compared bit for bit; `C32` by SNR against the fallback (the sums are formed in a
//! different order) and, for the block-size invariance, bit for bit between two runs of the
//! backend itself.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

mod common;

use common::{Rng, bits, c32_noise, chunks, cq15_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate, Resample, ResampleState};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::interp::{FirInterpState, FirInterpolate};
use sdrk_core::{C32, CQ15, Fallback, Q15};

const TAP_COUNTS: [usize; 9] = [1, 2, 3, 8, 31, 64, 127, 256, 300];

fn taps_f32(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.unit() / (n as f32).sqrt()).collect()
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
fn fir_c32_matches_the_fallback_within_the_contract() {
    let mut rng = Rng::new(20);
    for &n in &TAP_COUNTS {
        for len in [0usize, 1, 5, 63, 64, 65, 400, 1500] {
            let taps = taps_f32(&mut rng, n);
            let x = c32_noise(&mut rng, len, 1.0);
            let mut a =
                FirState::<C32, _, _>::new(taps.clone(), vec![C32::default(); n - 1]).unwrap();
            let mut b = a.clone();
            let (mut want, mut got) = (vec![C32::default(); len], vec![C32::default(); len]);
            // Two calls so the history is non-trivial in the second.
            let half = len / 2;
            Fallback.fir(&mut a, &x[..half], &mut want[..half]).unwrap();
            Fallback.fir(&mut a, &x[half..], &mut want[half..]).unwrap();
            Simd.fir(&mut b, &x[..half], &mut got[..half]).unwrap();
            Simd.fir(&mut b, &x[half..], &mut got[half..]).unwrap();
            if len > 0 {
                let snr = snr_db(&want, &got);
                assert!(snr > 120.0, "taps={n} len={len} snr={snr}");
            }
            assert_eq!(a.history(), b.history(), "history taps={n} len={len}");
        }
    }
}

#[test]
fn fir_cq15_is_bit_exact() {
    let mut rng = Rng::new(21);
    for &n in &TAP_COUNTS {
        for len in [0usize, 1, 5, 63, 64, 65, 400, 1500] {
            for extreme in [false, true] {
                let taps = taps_q15(&mut rng, n, extreme);
                let x = if extreme {
                    vec![
                        CQ15 {
                            re: Q15(-32768),
                            im: Q15(32767)
                        };
                        len
                    ]
                } else {
                    cq15_noise(&mut rng, len)
                };
                let mut a = FirState::<CQ15, _, _>::new(taps.clone(), vec![CQ15::default(); n - 1])
                    .unwrap();
                let mut b = a.clone();
                let (mut want, mut got) = (vec![CQ15::default(); len], vec![CQ15::default(); len]);
                let half = len / 2;
                Fallback.fir(&mut a, &x[..half], &mut want[..half]).unwrap();
                Fallback.fir(&mut a, &x[half..], &mut want[half..]).unwrap();
                Simd.fir(&mut b, &x[..half], &mut got[..half]).unwrap();
                Simd.fir(&mut b, &x[half..], &mut got[half..]).unwrap();
                assert_eq!(want, got, "taps={n} len={len} extreme={extreme}");
                assert_eq!(a.history(), b.history());
            }
        }
    }
}

#[test]
fn fir_c32_is_block_size_invariant() {
    let mut rng = Rng::new(22);
    for &n in &[1usize, 7, 64, 127, 300] {
        let taps = taps_f32(&mut rng, n);
        let len = 1200;
        let x = c32_noise(&mut rng, len, 1.0);
        let mut a = FirState::<C32, _, _>::new(taps.clone(), vec![C32::default(); n - 1]).unwrap();
        let mut b = a.clone();
        let mut whole = vec![C32::default(); len];
        Simd.fir(&mut a, &x, &mut whole).unwrap();
        let mut parts = vec![C32::default(); len];
        let mut at = 0;
        for c in chunks(&mut rng, len) {
            Simd.fir(&mut b, &x[at..at + c], &mut parts[at..at + c])
                .unwrap();
            at += c;
        }
        assert_eq!(bits(&whole), bits(&parts), "taps={n}");
        assert_eq!(a.history(), b.history());
    }
}

#[test]
fn fir_length_mismatch_is_an_error() {
    let mut s = FirState::<C32, _, _>::new(vec![1.0f32; 3], vec![C32::default(); 2]).unwrap();
    let mut out = [C32::default(); 3];
    assert_eq!(
        Fallback.fir(&mut s.clone(), &[C32::default(); 4], &mut out),
        Simd.fir(&mut s, &[C32::default(); 4], &mut out)
    );
}

#[test]
fn fir_decimate_matches_the_fallback() {
    let mut rng = Rng::new(23);
    for &n in &[1usize, 5, 64, 127, 300] {
        for factor in [1usize, 2, 3, 10, 33] {
            for room in [0usize, 1, 7, 100_000] {
                let len = 800;
                let taps = taps_f32(&mut rng, n);
                let x = c32_noise(&mut rng, len, 1.0);
                let mut a = FirDecimState::<C32, _, _>::new(
                    taps.clone(),
                    vec![C32::default(); n - 1],
                    factor,
                )
                .unwrap();
                let mut b = a.clone();
                let out_len = room.min(len);
                let (mut want, mut got) =
                    (vec![C32::default(); out_len], vec![C32::default(); out_len]);
                let pw = Fallback.fir_decimate(&mut a, &x, &mut want).unwrap();
                let pg = Simd.fir_decimate(&mut b, &x, &mut got).unwrap();
                assert_eq!(pw, pg, "taps={n} m={factor} room={room}");
                assert_eq!(a.countdown(), b.countdown());
                assert_eq!(a.history(), b.history());
                let k = pw.produced;
                if k > 0 {
                    let snr = snr_db(&want[..k], &got[..k]);
                    assert!(snr > 120.0, "taps={n} m={factor} snr={snr}");
                }
                // A second call continues where the first stopped.
                let rest = &x[pw.consumed..];
                let (mut want2, mut got2) = (vec![C32::default(); len], vec![C32::default(); len]);
                let pw2 = Fallback.fir_decimate(&mut a, rest, &mut want2).unwrap();
                let pg2 = Simd.fir_decimate(&mut b, rest, &mut got2).unwrap();
                assert_eq!(pw2, pg2);
                assert_eq!(a.countdown(), b.countdown());
                assert_eq!(a.history(), b.history());
            }
        }
    }
}

#[test]
fn fir_decimate_cq15_is_bit_exact() {
    let mut rng = Rng::new(24);
    for &n in &[1usize, 5, 64, 127, 300] {
        for factor in [1usize, 4, 10] {
            for room in [0usize, 3, 100_000] {
                let len = 700;
                let taps = taps_q15(&mut rng, n, n == 5);
                let x = cq15_noise(&mut rng, len);
                let mut a = FirDecimState::<CQ15, _, _>::new(
                    taps.clone(),
                    vec![CQ15::default(); n - 1],
                    factor,
                )
                .unwrap();
                let mut b = a.clone();
                let out_len = room.min(len);
                let (mut want, mut got) = (
                    vec![CQ15::default(); out_len],
                    vec![CQ15::default(); out_len],
                );
                let pw = Fallback.fir_decimate(&mut a, &x, &mut want).unwrap();
                let pg = Simd.fir_decimate(&mut b, &x, &mut got).unwrap();
                assert_eq!(pw, pg);
                assert_eq!(want, got, "taps={n} m={factor} room={room}");
                assert_eq!(a.countdown(), b.countdown());
                assert_eq!(a.history(), b.history());
            }
        }
    }
}

#[test]
fn fir_decimate_c32_is_block_size_invariant() {
    let mut rng = Rng::new(25);
    let (n, factor, len) = (127usize, 10usize, 3000usize);
    let taps = taps_f32(&mut rng, n);
    let x = c32_noise(&mut rng, len, 1.0);
    let mut a =
        FirDecimState::<C32, _, _>::new(taps.clone(), vec![C32::default(); n - 1], factor).unwrap();
    let mut b = a.clone();
    let mut whole = vec![C32::default(); len];
    let p = Simd.fir_decimate(&mut a, &x, &mut whole).unwrap();
    assert_eq!(p.consumed, len);
    let mut parts = Vec::new();
    let mut at = 0;
    for c in chunks(&mut rng, len) {
        let mut out = vec![C32::default(); c];
        let q = Simd.fir_decimate(&mut b, &x[at..at + c], &mut out).unwrap();
        assert_eq!(q.consumed, c);
        parts.extend_from_slice(&out[..q.produced]);
        at += c;
    }
    assert_eq!(bits(&whole[..p.produced]), bits(&parts));
}

fn resampler_taps(rng: &mut Rng, n: usize) -> Vec<f32> {
    taps_f32(rng, n)
}

#[test]
fn resample_and_interpolate_match_the_fallback() {
    let mut rng = Rng::new(26);
    for &(up, down) in &[
        (1usize, 1usize),
        (2, 1),
        (3, 1),
        (10, 1),
        (3, 2),
        (5, 7),
        (160, 147),
    ] {
        for &n in &[1usize, 4, 31, 64, 200] {
            for room in [0usize, 5, 100_000] {
                let len = 120;
                let hist = n.div_ceil(up);
                let taps = resampler_taps(&mut rng, n);
                let x = c32_noise(&mut rng, len, 1.0);
                let mut a = ResampleState::<C32, _, _>::new(
                    taps.clone(),
                    vec![C32::default(); hist],
                    up,
                    down,
                )
                .unwrap();
                let mut b = a.clone();
                let out_len = (len * up / down + 4).min(room.max(1) * 1000).min(3000);
                let out_len = if room == 0 {
                    0
                } else if room == 5 {
                    5
                } else {
                    out_len
                };
                let (mut want, mut got) =
                    (vec![C32::default(); out_len], vec![C32::default(); out_len]);
                let pw = Fallback.resample(&mut a, &x, &mut want).unwrap();
                let pg = Simd.resample(&mut b, &x, &mut got).unwrap();
                assert_eq!(pw, pg, "L={up} M={down} taps={n} room={room}");
                assert_eq!(a.phase(), b.phase());
                assert_eq!(a.history(), b.history());
                if pw.produced > 0 {
                    let snr = snr_db(&want[..pw.produced], &got[..pw.produced]);
                    assert!(snr > 120.0, "L={up} M={down} taps={n} snr={snr}");
                }
            }
        }
    }
    // The interpolator is the resampler with down = 1.
    for up in [2usize, 4, 10] {
        let n = 40;
        let taps = resampler_taps(&mut rng, n);
        let x = c32_noise(&mut rng, 100, 1.0);
        let mut a = FirInterpState::<C32, _, _>::new(
            taps.clone(),
            vec![C32::default(); n.div_ceil(up)],
            up,
        )
        .unwrap();
        let mut b = a.clone();
        let (mut want, mut got) = (
            vec![C32::default(); 100 * up],
            vec![C32::default(); 100 * up],
        );
        let pw = Fallback.fir_interpolate(&mut a, &x, &mut want).unwrap();
        let pg = Simd.fir_interpolate(&mut b, &x, &mut got).unwrap();
        assert_eq!(pw, pg);
        assert!(snr_db(&want, &got) > 120.0);
    }
}

#[test]
fn resample_and_interpolate_cq15_are_bit_exact() {
    let mut rng = Rng::new(27);
    for &(up, down) in &[(1usize, 1usize), (3, 1), (10, 1), (3, 2), (5, 7)] {
        for &n in &[1usize, 4, 31, 64] {
            let len = 100;
            let taps = taps_q15(&mut rng, n, false);
            let x = cq15_noise(&mut rng, len);
            let mut a = ResampleState::<CQ15, _, _>::new(
                taps.clone(),
                vec![CQ15::default(); n.div_ceil(up)],
                up,
                down,
            )
            .unwrap();
            let mut b = a.clone();
            let (mut want, mut got) = (vec![CQ15::default(); 2000], vec![CQ15::default(); 2000]);
            let pw = Fallback.resample(&mut a, &x, &mut want).unwrap();
            let pg = Simd.resample(&mut b, &x, &mut got).unwrap();
            assert_eq!(pw, pg);
            assert_eq!(want, got, "L={up} M={down} taps={n}");
            assert_eq!(a.phase(), b.phase());
        }
    }
    let up = 4;
    let taps = taps_q15(&mut rng, 30, false);
    let x = cq15_noise(&mut rng, 60);
    let mut a = FirInterpState::<CQ15, _, _>::new(
        taps.clone(),
        vec![CQ15::default(); 30usize.div_ceil(up)],
        up,
    )
    .unwrap();
    let mut b = a.clone();
    let (mut want, mut got) = (vec![CQ15::default(); 240], vec![CQ15::default(); 240]);
    Fallback.fir_interpolate(&mut a, &x, &mut want).unwrap();
    Simd.fir_interpolate(&mut b, &x, &mut got).unwrap();
    assert_eq!(want, got);
}
