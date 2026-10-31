//! `halfband_decimate`, `halfband_decimate_real` and `real_to_complex` against the fallback.
//!
//! `Q15` and `CQ15` are compared bit for bit (outputs, progress, phase and history); `f32` and
//! `C32` by SNR against the fallback (the sums are formed in a different order, with the
//! symmetric pre-add) and, for the block-size invariance, bit for bit between runs of the
//! backend itself. The filters cover the vector remainders (`N` from 3 to the longest one of
//! the backend and beyond), every input length and output room, chunked calls, the saturation
//! corners of `Q15` and the filters that run on the fallback (asymmetric taps, a sum of the
//! absolute taps above the exact range, more than the longest length).
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::type_complexity
)]

mod common;

use common::{LENGTHS, Rng, c32_noise, chunks, cq, cq15_noise, db};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::halfband::{
    HalfbandDecimate, HalfbandDecimateReal, HalfbandRealState, HalfbandState, RealToComplex,
    RealToComplexState,
};
use sdrk_core::{C32, CQ15, Fallback, Progress, Q15};

/// Filter lengths: the shortest, the vector remainders, the typical ones, around the longest
/// length of the backend (1023) and beyond it (the fallback).
const LENS: [usize; 14] = [
    3, 7, 11, 15, 23, 31, 47, 63, 127, 255, 515, 1023, 1027, 1203,
];

/// A symmetric halfband filter in `f32`: non-zero taps at the even indices and the centre.
fn taps_f32(rng: &mut Rng, n: usize) -> Vec<f32> {
    let m = (n - 1) / 2;
    let mut t = vec![0.0f32; n];
    for i in 0..=m / 2 {
        let v = rng.unit() / (n as f32).sqrt();
        t[2 * i] = v;
        t[n - 1 - 2 * i] = v;
    }
    t[m] = 0.5;
    t
}

/// `Q15` taps with a sum of the absolute values well inside the exact range of the backend.
fn taps_q15(rng: &mut Rng, n: usize) -> Vec<Q15> {
    let m = (n - 1) / 2;
    let scale = (n / 4).max(1) as i32;
    let mut t = vec![Q15(0); n];
    for i in 0..=m {
        t[2 * i] = Q15((i32::from(rng.next_u32() as i16) / scale) as i16);
    }
    t[m] = Q15(16384);
    t
}

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

fn cq15_extreme(rng: &mut Rng, n: usize) -> Vec<CQ15> {
    (0..n)
        .map(|i| match i % 5 {
            0 => cq(-32768, -32768),
            1 => cq(32767, -32768),
            _ => cq(rng.next_u32() as i16, rng.next_u32() as i16),
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

/// The room of the output of a call: none, a few samples, or enough for all.
fn room(len: usize, which: usize) -> usize {
    match which {
        0 => 0,
        1 => 5,
        _ => len / 2 + 2,
    }
}

/// Components of a sample as `f32`, for the SNR and the bit comparisons.
trait Flat: Copy + Default + PartialEq + core::fmt::Debug {
    fn flat(v: &[Self]) -> Vec<f32>;
}

impl Flat for C32 {
    fn flat(v: &[C32]) -> Vec<f32> {
        v.iter().flat_map(|c| [c.re, c.im]).collect()
    }
}

impl Flat for CQ15 {
    fn flat(v: &[CQ15]) -> Vec<f32> {
        v.iter()
            .flat_map(|c| [f32::from(c.re.0), f32::from(c.im.0)])
            .collect()
    }
}

impl Flat for f32 {
    fn flat(v: &[f32]) -> Vec<f32> {
        v.to_vec()
    }
}

impl Flat for Q15 {
    fn flat(v: &[Q15]) -> Vec<f32> {
        v.iter().map(|c| f32::from(c.0)).collect()
    }
}

/// The tests of one kernel.
macro_rules! kernel_tests {
    (
        $mod:ident, $t:ty, $tap:ty, $state_ty:ty, $new:expr, $call:ident,
        taps = $taps:expr, noise = $noise:expr, extreme = $extreme:expr, exact = $exact:expr
    ) => {
        mod $mod {
            use super::*;

            type St = $state_ty;

            fn state(taps: &[$tap]) -> St {
                let n = taps.len();
                let new: fn(Vec<$tap>, Vec<$t>) -> St = $new;
                new(taps.to_vec(), vec![<$t>::default(); n - 1])
            }

            /// Compares the backend with the fallback on the same input in chunks.
            fn compare(taps: &[$tap], x: &[$t], sizes: &[usize], which: usize, at: &str) {
                let (mut a, mut b) = (state(taps), state(taps));
                let mut pos = 0;
                let (mut all_want, mut all_got) = (Vec::<f32>::new(), Vec::<f32>::new());
                for (call, &len) in sizes.iter().enumerate() {
                    let chunk = &x[pos..pos + len];
                    pos += len;
                    let room = room(len, (which + call) % 3);
                    let (mut want, mut got) =
                        (vec![<$t>::default(); room], vec![<$t>::default(); room]);
                    let pw: Progress = Fallback.$call(&mut a, chunk, &mut want).unwrap();
                    let pg: Progress = Simd.$call(&mut b, chunk, &mut got).unwrap();
                    let at = format!("{at} call {call} len {len} room {room}");
                    assert_eq!(pw, pg, "{at}");
                    assert_eq!(a.history(), b.history(), "{at}");
                    assert_eq!(a.countdown(), b.countdown(), "{at}");
                    // Nothing beyond the outputs produced is written.
                    assert!(
                        got[pg.produced..].iter().all(|v| *v == <$t>::default()),
                        "{at}"
                    );
                    if $exact {
                        assert_eq!(&want[..pw.produced], &got[..pg.produced], "{at}");
                    } else {
                        all_want.extend(<$t as Flat>::flat(&want[..pw.produced]));
                        all_got.extend(<$t as Flat>::flat(&got[..pg.produced]));
                    }
                }
                // The SNR over the whole run: single outputs of a short call may cancel.
                if !all_want.is_empty() {
                    let s = snr(&all_want, &all_got);
                    assert!(s > 120.0, "{at} snr={s}");
                }
            }

            #[test]
            fn matches_the_fallback_for_every_filter_length() {
                let mut rng = Rng::new(701);
                for &n in &LENS {
                    let taps: Vec<$tap> = ($taps)(&mut rng, n);
                    let x: Vec<$t> = ($noise)(&mut rng, 700);
                    for which in 0..3 {
                        compare(&taps, &x, &[700], which, &format!("n={n}"));
                    }
                }
            }

            #[test]
            fn covers_every_input_length_and_chunking() {
                let mut rng = Rng::new(702);
                for n in [3usize, 11, 23, 47, 63, 131] {
                    let taps: Vec<$tap> = ($taps)(&mut rng, n);
                    for len in LENGTHS {
                        let x: Vec<$t> = ($noise)(&mut rng, len);
                        compare(&taps, &x, &[len], 2, &format!("n={n} len={len}"));
                    }
                    let x: Vec<$t> = ($noise)(&mut rng, 3000);
                    for round in 0..4 {
                        let sizes = chunks(&mut rng, x.len());
                        compare(&taps, &x, &sizes, round, &format!("n={n} chunked {round}"));
                    }
                }
            }

            #[test]
            fn is_block_size_invariant() {
                let mut rng = Rng::new(703);
                for n in [3usize, 23, 47, 131] {
                    let taps: Vec<$tap> = ($taps)(&mut rng, n);
                    let x: Vec<$t> = ($noise)(&mut rng, 1500);
                    let run = |sizes: &[usize]| {
                        let mut s = state(&taps);
                        let mut out = Vec::new();
                        let mut pos = 0;
                        for &len in sizes {
                            let mut y = vec![<$t>::default(); len / 2 + 2];
                            let p = Simd.$call(&mut s, &x[pos..pos + len], &mut y).unwrap();
                            out.extend_from_slice(&y[..p.produced]);
                            pos += len;
                        }
                        <$t as Flat>::flat(&out)
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    };
                    let whole = run(&[x.len()]);
                    for _ in 0..4 {
                        let sizes = chunks(&mut rng, x.len());
                        assert_eq!(whole, run(&sizes), "n={n}");
                    }
                }
            }

            #[test]
            fn saturation_corners_match_the_fallback() {
                let mut rng = Rng::new(704);
                for n in [3usize, 7, 23, 47, 131] {
                    let taps: Vec<$tap> = ($taps)(&mut rng, n);
                    let x: Vec<$t> = ($extreme)(&mut rng, 900);
                    compare(&taps, &x, &[900], 2, &format!("extreme n={n}"));
                    compare(
                        &taps,
                        &x,
                        &[100, 1, 299, 500],
                        0,
                        &format!("extreme chunked n={n}"),
                    );
                }
            }
        }
    };
}

kernel_tests!(
    halfband_c32, C32, f32, HalfbandState<C32, Vec<f32>, Vec<C32>>,
    |t, h| HalfbandState::new(t, h).unwrap(), halfband_decimate,
    taps = |r: &mut Rng, n| taps_f32(r, n),
    noise = |r: &mut Rng, n| c32_noise(r, n, 1.0),
    extreme = |r: &mut Rng, n| c32_noise(r, n, 1.0e3),
    exact = false
);
kernel_tests!(
    halfband_cq15, CQ15, Q15, HalfbandState<CQ15, Vec<Q15>, Vec<CQ15>>,
    |t, h| HalfbandState::new(t, h).unwrap(), halfband_decimate,
    taps = |r: &mut Rng, n| taps_q15(r, n),
    noise = |r: &mut Rng, n| cq15_noise(r, n),
    extreme = |r: &mut Rng, n| cq15_extreme(r, n),
    exact = true
);
kernel_tests!(
    halfband_real_f32, f32, f32, HalfbandRealState<f32, Vec<f32>, Vec<f32>>,
    |t, h| HalfbandRealState::new(t, h).unwrap(), halfband_decimate_real,
    taps = |r: &mut Rng, n| taps_f32(r, n),
    noise = |r: &mut Rng, n| f32_noise(r, n),
    extreme = |r: &mut Rng, n| f32_noise(r, n).iter().map(|v| v * 1.0e3).collect::<Vec<_>>(),
    exact = false
);
kernel_tests!(
    halfband_real_q15, Q15, Q15, HalfbandRealState<Q15, Vec<Q15>, Vec<Q15>>,
    |t, h| HalfbandRealState::new(t, h).unwrap(), halfband_decimate_real,
    taps = |r: &mut Rng, n| taps_q15(r, n),
    noise = |r: &mut Rng, n| q15_noise(r, n, false),
    extreme = |r: &mut Rng, n| q15_noise(r, n, true),
    exact = true
);

/// `Q15` filters at the edge of the exact range of the backend and over it.
#[test]
fn q15_extreme_taps_match_the_fallback() {
    let mut rng = Rng::new(705);
    for n in [3usize, 7, 23, 47] {
        let m = (n - 1) / 2;
        // Sum of the absolute taps 65535 (the bound of the exact `i32` sums), 65536 (fallback),
        // and the full-scale tap values.
        for (centre, other) in [
            (32767i16, -32768i16),
            (32767, 32767),
            (-32768, -32768),
            (0, 0),
        ] {
            let mut taps = vec![Q15(0); n];
            taps[m] = Q15(centre);
            taps[0] = Q15(other);
            let x = q15_noise(&mut rng, 600, true);
            let mk = || HalfbandRealState::new(taps.clone(), vec![Q15(0); n - 1]).unwrap();
            let (mut a, mut b) = (mk(), mk());
            let (mut want, mut got) = (vec![Q15(0); 305], vec![Q15(0); 305]);
            let pw = Fallback
                .halfband_decimate_real(&mut a, &x, &mut want)
                .unwrap();
            let pg = Simd.halfband_decimate_real(&mut b, &x, &mut got).unwrap();
            assert_eq!(pw, pg, "n={n} {centre} {other}");
            assert_eq!(want, got, "n={n} {centre} {other}");
            assert_eq!(a.history(), b.history());
            // All minimum samples against the extreme taps: the largest products.
            let x = vec![Q15(-32768); 400];
            let (mut a, mut b) = (mk(), mk());
            let (mut want, mut got) = (vec![Q15(0); 205], vec![Q15(0); 205]);
            Fallback
                .halfband_decimate_real(&mut a, &x, &mut want)
                .unwrap();
            Simd.halfband_decimate_real(&mut b, &x, &mut got).unwrap();
            assert_eq!(want, got, "min n={n} {centre} {other}");
        }
    }
}

/// Asymmetric `f32` taps run on the fallback: the same bits.
#[test]
fn asymmetric_f32_taps_run_on_the_fallback() {
    let mut rng = Rng::new(706);
    for n in [7usize, 23, 47] {
        let mut taps = taps_f32(&mut rng, n);
        taps[0] += 0.01;
        let x = f32_noise(&mut rng, 500);
        let mk = || HalfbandRealState::new(taps.clone(), vec![0.0f32; n - 1]).unwrap();
        let (mut a, mut b) = (mk(), mk());
        let (mut want, mut got) = (vec![0f32; 260], vec![0f32; 260]);
        Fallback
            .halfband_decimate_real(&mut a, &x, &mut want)
            .unwrap();
        Simd.halfband_decimate_real(&mut b, &x, &mut got).unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&want), bits(&got));
    }
}

/// A non-finite sample spoils exactly the outputs whose window holds it.
#[test]
fn a_non_finite_sample_stays_inside_its_windows() {
    let mut rng = Rng::new(707);
    for n in [3usize, 7, 23, 47] {
        let taps = taps_f32(&mut rng, n);
        let len = 3 * n + 40;
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for at in [0, 1, n / 2, n, n + 1, len - 2, len - 1] {
                let mut x = f32_noise(&mut rng, len);
                x[at] = bad;
                let mk = || HalfbandRealState::new(taps.clone(), vec![0.0f32; n - 1]).unwrap();
                let (mut a, mut b) = (mk(), mk());
                let (mut want, mut got) = (vec![0f32; len / 2 + 2], vec![0f32; len / 2 + 2]);
                let pw = Fallback
                    .halfband_decimate_real(&mut a, &x, &mut want)
                    .unwrap();
                let pg = Simd.halfband_decimate_real(&mut b, &x, &mut got).unwrap();
                assert_eq!(pw, pg);
                for (j, (w, g)) in want.iter().zip(&got).enumerate().take(pw.produced) {
                    assert_eq!(w.is_nan(), g.is_nan(), "n={n} at={at} j={j}");
                    assert_eq!(w.is_finite(), g.is_finite(), "n={n} at={at} j={j}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// real_to_complex

macro_rules! r2c_tests {
    (
        $mod:ident, $t:ty, $tap:ty, $in:ty, taps = $taps:expr, noise = $noise:expr,
        exact = $exact:expr
    ) => {
        mod $mod {
            use super::*;

            type St = RealToComplexState<$t, Vec<$tap>, Vec<$t>>;

            fn state(taps: &[$tap]) -> St {
                RealToComplexState::new(taps.to_vec(), vec![<$t>::default(); taps.len() - 1])
                    .unwrap()
            }

            fn compare(taps: &[$tap], x: &[$in], sizes: &[usize], which: usize, at: &str) {
                let (mut a, mut b) = (state(taps), state(taps));
                let mut pos = 0;
                let (mut all_want, mut all_got) = (Vec::<f32>::new(), Vec::<f32>::new());
                for (call, &len) in sizes.iter().enumerate() {
                    let chunk = &x[pos..pos + len];
                    pos += len;
                    let room = room(len, (which + call) % 3);
                    let (mut want, mut got) =
                        (vec![<$t>::default(); room], vec![<$t>::default(); room]);
                    let pw = Fallback.real_to_complex(&mut a, chunk, &mut want).unwrap();
                    let pg = Simd.real_to_complex(&mut b, chunk, &mut got).unwrap();
                    let at = format!("{at} call {call} len {len} room {room}");
                    assert_eq!(pw, pg, "{at}");
                    assert_eq!(a.history(), b.history(), "{at}");
                    assert_eq!(a.n_mod_4(), b.n_mod_4(), "{at}");
                    assert!(
                        got[pg.produced..].iter().all(|v| *v == <$t>::default()),
                        "{at}"
                    );
                    if $exact {
                        assert_eq!(&want[..pw.produced], &got[..pg.produced], "{at}");
                    } else {
                        all_want.extend(<$t as Flat>::flat(&want[..pw.produced]));
                        all_got.extend(<$t as Flat>::flat(&got[..pg.produced]));
                    }
                }
                // The SNR over the whole run: single outputs of a short call may cancel.
                if !all_want.is_empty() {
                    let s = snr(&all_want, &all_got);
                    assert!(s > 120.0, "{at} snr={s}");
                }
            }

            #[test]
            fn matches_the_fallback_for_every_filter_length() {
                let mut rng = Rng::new(711);
                for &n in &LENS {
                    let taps: Vec<$tap> = ($taps)(&mut rng, n);
                    let x: Vec<$in> = ($noise)(&mut rng, 700, false);
                    for which in 0..3 {
                        compare(&taps, &x, &[700], which, &format!("n={n}"));
                    }
                }
            }

            #[test]
            fn covers_every_input_length_phase_and_chunking() {
                let mut rng = Rng::new(712);
                for n in [3usize, 11, 23, 47, 63, 131] {
                    let taps: Vec<$tap> = ($taps)(&mut rng, n);
                    for len in LENGTHS {
                        let x: Vec<$in> = ($noise)(&mut rng, len, false);
                        compare(&taps, &x, &[len], 2, &format!("n={n} len={len}"));
                    }
                    let x: Vec<$in> = ($noise)(&mut rng, 3000, false);
                    for round in 0..6 {
                        let sizes = chunks(&mut rng, x.len());
                        compare(&taps, &x, &sizes, round, &format!("n={n} chunked {round}"));
                    }
                }
            }

            #[test]
            fn extremes_match_the_fallback() {
                let mut rng = Rng::new(713);
                for n in [3usize, 7, 23, 47, 131] {
                    let taps: Vec<$tap> = ($taps)(&mut rng, n);
                    let x: Vec<$in> = ($noise)(&mut rng, 900, true);
                    compare(&taps, &x, &[900], 2, &format!("extreme n={n}"));
                    compare(
                        &taps,
                        &x,
                        &[1, 100, 3, 296, 500],
                        0,
                        &format!("chunked n={n}"),
                    );
                }
            }

            #[test]
            fn is_block_size_invariant() {
                let mut rng = Rng::new(714);
                for n in [3usize, 23, 47, 131] {
                    let taps: Vec<$tap> = ($taps)(&mut rng, n);
                    let x: Vec<$in> = ($noise)(&mut rng, 1500, false);
                    let run = |sizes: &[usize]| {
                        let mut s = state(&taps);
                        let mut out = Vec::new();
                        let mut pos = 0;
                        for &len in sizes {
                            let mut y = vec![<$t>::default(); len / 2 + 2];
                            let p = Simd
                                .real_to_complex(&mut s, &x[pos..pos + len], &mut y)
                                .unwrap();
                            out.extend_from_slice(&y[..p.produced]);
                            pos += len;
                        }
                        <$t as Flat>::flat(&out)
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    };
                    let whole = run(&[x.len()]);
                    for _ in 0..4 {
                        let sizes = chunks(&mut rng, x.len());
                        assert_eq!(whole, run(&sizes), "n={n}");
                    }
                }
            }
        }
    };
}

r2c_tests!(
    r2c_c32,
    C32,
    f32,
    f32,
    taps = |r: &mut Rng, n| taps_f32(r, n),
    noise = |r: &mut Rng, n, extreme: bool| {
        f32_noise(r, n)
            .iter()
            .map(|v| if extreme { v * 1.0e3 } else { *v })
            .collect::<Vec<_>>()
    },
    exact = false
);
r2c_tests!(
    r2c_cq15,
    CQ15,
    Q15,
    Q15,
    taps = |r: &mut Rng, n| taps_q15(r, n),
    noise = |r: &mut Rng, n, extreme: bool| q15_noise(r, n, extreme),
    exact = true
);

/// The saturating negation of the shift: the sample `-32768` at every phase.
#[test]
fn r2c_cq15_saturates_the_negation_like_the_fallback() {
    let mut rng = Rng::new(715);
    for n in [3usize, 7, 23, 47] {
        let taps = taps_q15(&mut rng, n);
        for fill in [-32768i16, 32767, -1] {
            for phase in 0..4 {
                let mk = || {
                    let mut s = RealToComplexState::<CQ15, _, _>::new(
                        taps.clone(),
                        vec![CQ15::default(); n - 1],
                    )
                    .unwrap();
                    // Advance the phase (and the history) with a few samples.
                    let mut sink = vec![CQ15::default(); 4];
                    Fallback
                        .real_to_complex(&mut s, &vec![Q15(fill); phase], &mut sink)
                        .unwrap();
                    s
                };
                let (mut a, mut b) = (mk(), mk());
                let x = vec![Q15(fill); 300];
                let (mut want, mut got) = (vec![CQ15::default(); 160], vec![CQ15::default(); 160]);
                let pw = Fallback.real_to_complex(&mut a, &x, &mut want).unwrap();
                let pg = Simd.real_to_complex(&mut b, &x, &mut got).unwrap();
                assert_eq!(pw, pg);
                assert_eq!(want, got, "n={n} fill={fill} phase={phase}");
                assert_eq!(a.history(), b.history());
            }
        }
    }
}
