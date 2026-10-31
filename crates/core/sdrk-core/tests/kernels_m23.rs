//! Tests of the M23 kernels of `sdrk-core`: `halfband_decimate`, `halfband_decimate_real` and
//! `real_to_complex`: construction rules, timing, hand-computed values, equality with
//! `fir_decimate` with factor 2, the definition of `real_to_complex`, the phase across calls,
//! the saturating negation, the output-space rules of `Progress`, the extremes and thread safety.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::float_cmp,
    clippy::type_complexity,
    clippy::items_after_statements
)]

use sdrk_core::kernels::decim::{FirDecimState, FirDecimate};
use sdrk_core::kernels::halfband::{
    HalfbandDecimate, HalfbandDecimateReal, HalfbandParts, HalfbandRealState, HalfbandState,
    RealToComplex, RealToComplexParts, RealToComplexState,
};
use sdrk_core::{C32, CQ15, Delay, Error, Fallback, Progress, Q15, Ratio, Sample, Timing};
use sdrk_device::{ErrorKind, SdrkError};

/// A small deterministic generator (no dependency on the testkit, R8.7).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
    /// Uniform in [-scale, scale).
    fn unit(&mut self, scale: f32) -> f32 {
        (self.next() as f32 / (1u32 << 31) as f32 - 1.0) * scale
    }
}

/// Windowed-sinc halfband taps in f64: zero at the even offsets, `0.5` at the centre.
fn taps_f64(n: usize) -> Vec<f64> {
    let m = (n - 1) / 2;
    (0..n)
        .map(|k| {
            let d = k.abs_diff(m);
            if d == 0 {
                0.5
            } else if d % 2 == 0 {
                0.0
            } else {
                let x = d as f64;
                let w = 0.5 + 0.5 * (core::f64::consts::PI * x / (m as f64 + 1.0)).cos();
                (core::f64::consts::FRAC_PI_2 * x).sin() / (core::f64::consts::PI * x) * w
            }
        })
        .collect()
}

fn taps_f32(n: usize) -> Vec<f32> {
    taps_f64(n).iter().map(|&v| v as f32).collect()
}

fn taps_q15(n: usize) -> Vec<Q15> {
    taps_f64(n)
        .iter()
        .map(|&v| Q15((v * 32768.0 + 0.5).floor().clamp(-32768.0, 32767.0) as i16))
        .collect()
}

/// `fir_decimate` taps: the same taps are accepted by a plain `FirDecimState`.
fn rand_c32(rng: &mut Lcg, n: usize, scale: f32) -> Vec<C32> {
    (0..n)
        .map(|_| C32::new(rng.unit(scale), rng.unit(scale)))
        .collect()
}

fn rand_cq15(rng: &mut Lcg, n: usize) -> Vec<CQ15> {
    (0..n)
        .map(|_| CQ15 {
            re: Q15(rng.next() as i16),
            im: Q15(rng.next() as i16),
        })
        .collect()
}

fn rand_q15(rng: &mut Lcg, n: usize) -> Vec<Q15> {
    (0..n).map(|_| Q15(rng.next() as i16)).collect()
}

/// Drives `call` over `input` in random chunks with random (often tiny) output buffers, resubmitting
/// what was not consumed, like a block does. Returns all outputs.
fn drive<I: Copy, O: Sample>(
    input: &[I],
    seed: u64,
    mut call: impl FnMut(&[I], &mut [O]) -> Progress,
) -> Vec<O> {
    let mut rng = Lcg(seed);
    let mut out = Vec::new();
    let mut pos = 0;
    let caps = [1usize, 2, 3, 5, 8, 17, 64, 300];
    while pos < input.len() {
        let chunk = (rng.below(150) as usize + 1).min(input.len() - pos);
        let mut sub = &input[pos..pos + chunk];
        while !sub.is_empty() {
            let cap = caps[rng.below(caps.len() as u32) as usize];
            let mut buf = vec![O::ZERO; cap];
            let p = call(sub, &mut buf);
            assert!(p.produced <= cap && p.consumed <= sub.len());
            out.extend_from_slice(&buf[..p.produced]);
            sub = &sub[p.consumed..];
        }
        pos += chunk;
    }
    out
}

/// Runs `call` in one piece with an output buffer of the largest possible size.
fn whole<I: Copy, O: Sample>(
    input: &[I],
    mut call: impl FnMut(&[I], &mut [O]) -> Progress,
) -> Vec<O> {
    let mut buf = vec![O::ZERO; input.len() / 2 + 2];
    let p = call(input, &mut buf);
    assert_eq!(p.consumed, input.len());
    buf.truncate(p.produced);
    buf
}

fn c32_bits(v: &[C32]) -> Vec<(u32, u32)> {
    v.iter().map(|s| (s.re.to_bits(), s.im.to_bits())).collect()
}

const LENS: [usize; 6] = [3, 7, 11, 23, 47, 63];

#[test]
fn construction_rules() {
    let zero_c = |n: usize| vec![C32::default(); n];
    // N mod 4 != 3, N < 3.
    for n in [0usize, 1, 2, 4, 5, 6, 8, 9, 10, 13] {
        let taps = vec![0.0f32; n];
        let e =
            HalfbandState::<C32, _, _>::new(taps.clone(), zero_c(n.saturating_sub(1))).unwrap_err();
        assert!(matches!(e, Error::InvalidParameter(_)), "{n}: {e:?}");
        assert_eq!(e.kind(), ErrorKind::InvalidArgument);
        assert!(RealToComplexState::<C32, _, _>::new(taps, zero_c(n.saturating_sub(1))).is_err());
    }
    // A non-zero tap at an even offset from the centre (N = 7: m = 3, offsets 2 -> taps 1 and 5).
    for bad in [1usize, 5] {
        let mut t = taps_f32(7);
        t[bad] = 1e-30;
        assert!(matches!(
            HalfbandState::<C32, _, _>::new(t.clone(), zero_c(6)),
            Err(Error::InvalidParameter(_))
        ));
        assert!(matches!(
            HalfbandRealState::<f32, _, _>::new(t.clone(), vec![0.0; 6]),
            Err(Error::InvalidParameter(_))
        ));
        assert!(RealToComplexState::<C32, _, _>::new(t, zero_c(6)).is_err());
    }
    // The outermost taps (even index) and the odd offsets may be non-zero; a tap of -0.0 is zero.
    let mut t = taps_f32(11);
    t[1] = -0.0;
    assert!(HalfbandState::<C32, _, _>::new(t, zero_c(10)).is_ok());
    let mut t = taps_q15(11);
    t[3] = Q15(1);
    assert!(HalfbandRealState::<Q15, _, _>::new(t, vec![Q15(0); 10]).is_err());
    // The history length must be N - 1.
    for got in [0usize, 21, 23] {
        let r = HalfbandState::<CQ15, _, _>::new(taps_q15(23), vec![CQ15::default(); got]);
        assert_eq!(r.unwrap_err(), Error::LengthMismatch { expected: 22, got });
    }
    // A history of the right length is zeroed.
    let s = HalfbandRealState::<f32, _, _>::new(taps_f32(7), vec![1.0f32; 6]).unwrap();
    assert!(s.history().iter().all(|&v| v == 0.0));
    assert_eq!((s.len(), s.is_empty(), s.taps().len()), (7, false, 7));
}

#[test]
fn timing_is_half_rate_and_group_delay() {
    for n in LENS {
        let s = HalfbandState::<C32, _, _>::new(taps_f32(n), vec![C32::default(); n - 1]).unwrap();
        assert_eq!(s.ratio(), Ratio::new(1, 2).unwrap());
        assert_eq!(s.group_delay(), Delay::new(n as u64 - 1, 2).unwrap());
        let s = HalfbandRealState::<Q15, _, _>::new(taps_q15(n), vec![Q15(0); n - 1]).unwrap();
        assert_eq!(s.ratio(), Ratio::new(1, 2).unwrap());
        assert_eq!(s.group_delay().num(), n as u64 - 1);
        let s = RealToComplexState::<CQ15, _, _>::new(taps_q15(n), vec![CQ15::default(); n - 1])
            .unwrap();
        assert_eq!(s.ratio(), Ratio::new(1, 2).unwrap());
        assert_eq!(s.group_delay().den(), 2);
    }
}

#[test]
fn hand_computed_values() {
    // N = 7, h = [a, 0, b, c, b, 0, a]: an impulse at the input gives the non-zero taps at the
    // even output positions.
    let h = [0.125f32, 0.0, -0.25, 0.5, -0.25, 0.0, 0.125];
    let mut s = HalfbandRealState::<f32, _, _>::new(h, [0.0f32; 6]).unwrap();
    let mut x = [0.0f32; 16];
    x[0] = 1.0;
    let mut y = [0.0f32; 8];
    let p = Fallback.halfband_decimate_real(&mut s, &x, &mut y).unwrap();
    assert_eq!((p.consumed, p.produced), (16, 8));
    // y[m] = h[2 m] (the impulse response sampled at the even positions).
    assert_eq!(y, [0.125, -0.25, -0.25, 0.125, 0.0, 0.0, 0.0, 0.0]);
    // The odd positions are never output, so an impulse at the odd input sample 1 shows h[2m - 1].
    s.reset();
    x[0] = 0.0;
    x[1] = 1.0;
    Fallback.halfband_decimate_real(&mut s, &x, &mut y).unwrap();
    assert_eq!(y, [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

/// `fir_decimate` by 2 of `x` with the halfband taps `t`, summed as the halfband fallback does:
/// the even taps in ascending order, the centre tap last (D93). Two dense filters give exactly
/// that: the even taps alone, then the centre tap alone, added as `f32` (the omitted zero taps
/// change no bit of a finite sum).
fn fir_decimate_centre_last(t: &[f32], x: &[C32]) -> Vec<C32> {
    let n = t.len();
    let m = (n - 1) / 2;
    let mut even = t.to_vec();
    even[m] = 0.0;
    let mut centre = vec![0.0f32; n];
    centre[m] = t[m];
    let mut a = FirDecimState::<C32, _, _>::new(even, vec![C32::default(); n - 1], 2).unwrap();
    let mut b = FirDecimState::<C32, _, _>::new(centre, vec![C32::default(); n - 1], 2).unwrap();
    let ya = whole(x, |i, o| Fallback.fir_decimate(&mut a, i, o).unwrap());
    let yb = whole(x, |i, o| Fallback.fir_decimate(&mut b, i, o).unwrap());
    ya.iter()
        .zip(&yb)
        .map(|(p, q)| C32::new(p.re + q.re, p.im + q.im))
        .collect()
}

#[test]
fn halfband_equals_fir_decimate_by_two() {
    let mut rng = Lcg(23);
    for n in LENS {
        let x = rand_c32(&mut rng, 777, 1.0);
        let t = taps_f32(n);
        let mut b =
            HalfbandState::<C32, _, _>::new(t.clone(), vec![C32::default(); n - 1]).unwrap();
        let want = fir_decimate_centre_last(&t, &x);
        let got = whole(&x, |i, o| Fallback.halfband_decimate(&mut b, i, o).unwrap());
        // Finite inputs: the omitted zero taps change no bit (a zero result compares equal).
        assert_eq!(want, got, "n {n}");
        // Real input as the real part of a complex stream.
        let xr: Vec<f32> = x.iter().map(|s| s.re).collect();
        let mut r = HalfbandRealState::<f32, _, _>::new(t.clone(), vec![0.0; n - 1]).unwrap();
        let yr = whole(&xr, |i, o| {
            Fallback.halfband_decimate_real(&mut r, i, o).unwrap()
        });
        assert_eq!(yr, want.iter().map(|s| s.re).collect::<Vec<_>>());

        // Q15: bit for bit, including the saturating corners.
        let xq = rand_cq15(&mut rng, 777);
        let tq = taps_q15(n);
        let mut a =
            FirDecimState::<CQ15, _, _>::new(tq.clone(), vec![CQ15::default(); n - 1], 2).unwrap();
        let mut b =
            HalfbandState::<CQ15, _, _>::new(tq.clone(), vec![CQ15::default(); n - 1]).unwrap();
        let want = whole(&xq, |i, o| Fallback.fir_decimate(&mut a, i, o).unwrap());
        let got = whole(&xq, |i, o| {
            Fallback.halfband_decimate(&mut b, i, o).unwrap()
        });
        assert_eq!(want, got, "n {n}");
        let xr: Vec<Q15> = xq.iter().map(|s| s.re).collect();
        let mut r = HalfbandRealState::<Q15, _, _>::new(tq, vec![Q15(0); n - 1]).unwrap();
        let yr = whole(&xr, |i, o| {
            Fallback.halfband_decimate_real(&mut r, i, o).unwrap()
        });
        assert_eq!(yr, want.iter().map(|s| s.re).collect::<Vec<_>>());
    }
}

/// A halfband tap vector of `n` taps whose non-zero taps (the even offsets and the centre) have
/// the absolute values `mags` in order, with alternating signs, the rest zero.
fn taps_with_abs_sum(n: usize, total: i32) -> Vec<Q15> {
    let m = (n - 1) / 2;
    let idx: Vec<usize> = (0..n).filter(|k| k % 2 == 0 || *k == m).collect();
    let mut taps = vec![Q15(0); n];
    let each = total / idx.len() as i32;
    let extra = total - each * idx.len() as i32;
    for (j, k) in idx.iter().enumerate() {
        let mag = each + i32::from((j as i32) < extra);
        taps[*k] = Q15(if j % 2 == 0 { mag } else { -mag } as i16);
    }
    taps
}

#[test]
fn q15_sums_are_exact_at_the_edge_of_the_narrow_accumulator() {
    // The fallback sums a group in `i32` while the absolute tap sum is at most 65535 and in `i64`
    // above; both must equal the `i64` sums of `fir_decimate` (checked with extreme samples).
    let mut rng = Lcg(2331);
    for n in [11usize, 23, 47, 63] {
        for total in [30000, 65534, 65535, 65536, 65537, 90000] {
            let taps = taps_with_abs_sum(n, total);
            for pattern in 0..4 {
                let x: Vec<CQ15> = (0..500)
                    .map(|i| {
                        let v = |r: &mut Lcg| match pattern {
                            0 => Q15(i16::MIN),
                            1 => Q15(i16::MAX),
                            2 => Q15(if (i / 3) % 2 == 0 { i16::MIN } else { i16::MAX }),
                            _ => Q15(if r.next() & 1 == 0 {
                                i16::MIN
                            } else {
                                i16::MAX
                            }),
                        };
                        CQ15 {
                            re: v(&mut rng),
                            im: v(&mut rng),
                        }
                    })
                    .collect();
                let mut a =
                    FirDecimState::<CQ15, _, _>::new(taps.clone(), vec![CQ15::default(); n - 1], 2)
                        .unwrap();
                let mut b =
                    HalfbandState::<CQ15, _, _>::new(taps.clone(), vec![CQ15::default(); n - 1])
                        .unwrap();
                let want = whole(&x, |i, o| Fallback.fir_decimate(&mut a, i, o).unwrap());
                let got = whole(&x, |i, o| Fallback.halfband_decimate(&mut b, i, o).unwrap());
                assert_eq!(want, got, "n {n} total {total} pattern {pattern}");
                let xr: Vec<Q15> = x.iter().map(|s| s.re).collect();
                let mut r =
                    HalfbandRealState::<Q15, _, _>::new(taps.clone(), vec![Q15(0); n - 1]).unwrap();
                let yr = whole(&xr, |i, o| {
                    Fallback.halfband_decimate_real(&mut r, i, o).unwrap()
                });
                assert_eq!(yr, want.iter().map(|s| s.re).collect::<Vec<_>>());
                // real_to_complex: the explicit shift followed by the halfband decimator.
                let mut c = RealToComplexState::<CQ15, _, _>::new(
                    taps.clone(),
                    vec![CQ15::default(); n - 1],
                )
                .unwrap();
                let mut d =
                    HalfbandState::<CQ15, _, _>::new(taps.clone(), vec![CQ15::default(); n - 1])
                        .unwrap();
                let got = whole(&xr, |i, o| Fallback.real_to_complex(&mut c, i, o).unwrap());
                let shifted = shift_cq15(&xr, 0);
                let want = whole(&shifted, |i, o| {
                    Fallback.halfband_decimate(&mut d, i, o).unwrap()
                });
                assert_eq!(want, got, "r2c n {n} total {total} pattern {pattern}");
            }
        }
    }
}

#[test]
fn filters_longer_than_the_group_limit_are_computed_per_output() {
    // Above 511 taps the fallback does not group; the result is the same as `fir_decimate`.
    let mut rng = Lcg(515);
    for n in [511usize, 515, 1023] {
        let x = rand_c32(&mut rng, 2300, 1.0);
        let t = taps_f32(n);
        let mut b =
            HalfbandState::<C32, _, _>::new(t.clone(), vec![C32::default(); n - 1]).unwrap();
        let want = fir_decimate_centre_last(&t, &x);
        let got = whole(&x, |i, o| Fallback.halfband_decimate(&mut b, i, o).unwrap());
        assert_eq!(want, got, "n {n}");
        let xq = rand_cq15(&mut rng, 2300);
        let tq = taps_q15(n);
        let mut a =
            FirDecimState::<CQ15, _, _>::new(tq.clone(), vec![CQ15::default(); n - 1], 2).unwrap();
        let mut b = HalfbandState::<CQ15, _, _>::new(tq, vec![CQ15::default(); n - 1]).unwrap();
        let want = whole(&xq, |i, o| Fallback.fir_decimate(&mut a, i, o).unwrap());
        let got = whole(&xq, |i, o| {
            Fallback.halfband_decimate(&mut b, i, o).unwrap()
        });
        assert_eq!(want, got, "n {n}");
    }
}

#[test]
fn chunking_and_output_space_do_not_change_the_bits() {
    let mut rng = Lcg(5);
    for n in [3usize, 23, 63] {
        let x = rand_c32(&mut rng, 2000, 1.0);
        let t = taps_f32(n);
        let mut s =
            HalfbandState::<C32, _, _>::new(t.clone(), vec![C32::default(); n - 1]).unwrap();
        let want = whole(&x, |i, o| Fallback.halfband_decimate(&mut s, i, o).unwrap());
        for seed in 0..20 {
            let mut s =
                HalfbandState::<C32, _, _>::new(t.clone(), vec![C32::default(); n - 1]).unwrap();
            let got = drive(&x, seed, |i, o| {
                Fallback.halfband_decimate(&mut s, i, o).unwrap()
            });
            assert_eq!(c32_bits(&want), c32_bits(&got), "n {n} seed {seed}");
        }
        let xq = rand_q15(&mut rng, 2000);
        let tq = taps_q15(n);
        let mut s = HalfbandRealState::<Q15, _, _>::new(tq.clone(), vec![Q15(0); n - 1]).unwrap();
        let want = whole(&xq, |i, o| {
            Fallback.halfband_decimate_real(&mut s, i, o).unwrap()
        });
        for seed in 0..20 {
            let mut s =
                HalfbandRealState::<Q15, _, _>::new(tq.clone(), vec![Q15(0); n - 1]).unwrap();
            let got = drive(&xq, seed, |i, o| {
                Fallback.halfband_decimate_real(&mut s, i, o).unwrap()
            });
            assert_eq!(want, got, "n {n} seed {seed}");
        }
    }
}

#[test]
fn consumption_stops_before_the_next_producing_sample() {
    let h = taps_f32(7);
    let mut s = HalfbandRealState::<f32, _, _>::new(h, vec![0.0f32; 6]).unwrap();
    let x = [1.0f32; 10];
    // Output space for one sample: samples 0 (produces) and 1 (skipped) are consumed, sample 2 would
    // produce a second output.
    let mut y = [0.0f32; 1];
    let p = Fallback.halfband_decimate_real(&mut s, &x, &mut y).unwrap();
    assert_eq!((p.consumed, p.produced), (2, 1));
    assert_eq!(s.countdown(), 0);
    assert_eq!(s.max_input_for_output(3), 6);
    // No output space at all.
    let p = Fallback
        .halfband_decimate_real(&mut s, &x, &mut [])
        .unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    // An odd number of samples leaves the phase at 1.
    let mut y = [0.0f32; 8];
    let p = Fallback
        .halfband_decimate_real(&mut s, &x[..5], &mut y)
        .unwrap();
    assert_eq!((p.consumed, p.produced), (5, 3));
    assert_eq!(s.countdown(), 1);
    assert_eq!(s.max_input_for_output(0), 1);
    let p = Fallback
        .halfband_decimate_real(&mut s, &x[..1], &mut y)
        .unwrap();
    assert_eq!((p.consumed, p.produced), (1, 0));
    assert_eq!(s.countdown(), 0);
    // Empty input.
    let p = Fallback
        .halfband_decimate_real(&mut s, &[], &mut y)
        .unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
}

/// `u[n] = x[n] (-j)^n` by the formula on complex numbers, with the saturating negation for Q15.
fn shift_c32(x: &[f32], start: usize) -> Vec<C32> {
    x.iter()
        .enumerate()
        .map(|(i, &v)| match (start + i) % 4 {
            0 => C32::new(v, 0.0),
            1 => C32::new(0.0, -v),
            2 => C32::new(-v, 0.0),
            _ => C32::new(0.0, v),
        })
        .collect()
}

fn shift_cq15(x: &[Q15], start: usize) -> Vec<CQ15> {
    let neg = |v: Q15| Q15(v.0.saturating_neg());
    x.iter()
        .enumerate()
        .map(|(i, &v)| match (start + i) % 4 {
            0 => CQ15 { re: v, im: Q15(0) },
            1 => CQ15 {
                re: Q15(0),
                im: neg(v),
            },
            2 => CQ15 {
                re: neg(v),
                im: Q15(0),
            },
            _ => CQ15 { re: Q15(0), im: v },
        })
        .collect()
}

#[test]
fn real_to_complex_is_the_shift_followed_by_the_halfband_decimator() {
    let mut rng = Lcg(77);
    for n in LENS {
        // f32
        let x: Vec<f32> = (0..1500).map(|_| rng.unit(1.0)).collect();
        let t = taps_f32(n);
        let u = shift_c32(&x, 0);
        let mut h =
            HalfbandState::<C32, _, _>::new(t.clone(), vec![C32::default(); n - 1]).unwrap();
        let want = whole(&u, |i, o| Fallback.halfband_decimate(&mut h, i, o).unwrap());
        let mut s =
            RealToComplexState::<C32, _, _>::new(t.clone(), vec![C32::default(); n - 1]).unwrap();
        let got = whole(&x, |i, o| Fallback.real_to_complex(&mut s, i, o).unwrap());
        // The same bits (the zero components of u are exact zeros).
        assert_eq!(c32_bits(&want), c32_bits(&got), "n {n}");
        assert_eq!(s.history(), h.history(), "n {n}");
        for seed in 0..20 {
            let mut s =
                RealToComplexState::<C32, _, _>::new(t.clone(), vec![C32::default(); n - 1])
                    .unwrap();
            let got = drive(&x, seed, |i, o| {
                Fallback.real_to_complex(&mut s, i, o).unwrap()
            });
            assert_eq!(c32_bits(&want), c32_bits(&got), "n {n} seed {seed}");
        }

        // CQ15, with the corner -32768 sprinkled in.
        let mut xq = rand_q15(&mut rng, 1500);
        for k in (3..1500).step_by(7) {
            xq[k] = Q15(-32768);
        }
        let tq = taps_q15(n);
        let u = shift_cq15(&xq, 0);
        let mut h =
            HalfbandState::<CQ15, _, _>::new(tq.clone(), vec![CQ15::default(); n - 1]).unwrap();
        let want = whole(&u, |i, o| Fallback.halfband_decimate(&mut h, i, o).unwrap());
        let mut s = RealToComplexState::<CQ15, _, _>::new(tq.clone(), vec![CQ15::default(); n - 1])
            .unwrap();
        let got = whole(&xq, |i, o| Fallback.real_to_complex(&mut s, i, o).unwrap());
        assert_eq!(want, got, "n {n}");
        assert_eq!(s.history(), h.history());
        for seed in 0..20 {
            let mut s =
                RealToComplexState::<CQ15, _, _>::new(tq.clone(), vec![CQ15::default(); n - 1])
                    .unwrap();
            let got = drive(&xq, seed, |i, o| {
                Fallback.real_to_complex(&mut s, i, o).unwrap()
            });
            assert_eq!(want, got, "n {n} seed {seed}");
        }
    }
}

#[test]
fn the_negation_saturates_in_q15() {
    // N = 3, h = [0, 32767, 0] (the centre tap about 1): y[m] = u[2 m - 1] * h[1] only.
    let h = [Q15(0), Q15(32767), Q15(0)];
    let mut s = RealToComplexState::<CQ15, _, _>::new(h, [CQ15::default(); 2]).unwrap();
    // x[1] = -32768 is at n mod 4 = 1: u[1] = (0, -(-32768)) = (0, 32767).
    // x[3] = -32768 is at n mod 4 = 3: u[3] = (0, -32768).
    let x = [Q15(5), Q15(-32768), Q15(7), Q15(-32768), Q15(0), Q15(0)];
    let mut y = [CQ15::default(); 3];
    let p = Fallback.real_to_complex(&mut s, &x, &mut y).unwrap();
    assert_eq!((p.consumed, p.produced), (6, 3));
    let mul = |a: i16, b: i16| ((i32::from(a) * i32::from(b) + (1 << 14)) >> 15) as i16;
    assert_eq!(y[0], CQ15::default());
    assert_eq!(y[1].re, Q15(0));
    assert_eq!(y[1].im, Q15(mul(32767, 32767)));
    assert_eq!(y[2].im, Q15(mul(-32768, 32767)));
    // x[2] = -32768 would be at n mod 4 = 2: u[2] = (32767, 0) (re part, taps at even index).
    let h = [Q15(32767), Q15(0), Q15(0), Q15(0), Q15(0), Q15(0), Q15(0)];
    let mut s = RealToComplexState::<CQ15, _, _>::new(h, [CQ15::default(); 6]).unwrap();
    let x = [Q15(-32768); 5];
    let mut y = [CQ15::default(); 3];
    Fallback.real_to_complex(&mut s, &x, &mut y).unwrap();
    // y[m] = h[0] u[2 m]: u[0] = -32768, u[2] = 32767, u[4] = -32768 (real parts).
    assert_eq!(y[0].re, Q15(mul(-32768, 32767)));
    assert_eq!(y[1].re, Q15(mul(32767, 32767)));
    assert_eq!(y[2].re, Q15(mul(-32768, 32767)));
    assert!(y.iter().all(|v| v.im == Q15(0)));
}

#[test]
fn the_phase_continues_across_calls_and_restarts_at_reset() {
    let mut rng = Lcg(3);
    let n = 23;
    let x: Vec<f32> = (0..400).map(|_| rng.unit(1.0)).collect();
    let t = taps_f32(n);
    let mut a =
        RealToComplexState::<C32, _, _>::new(t.clone(), vec![C32::default(); n - 1]).unwrap();
    let want = whole(&x, |i, o| Fallback.real_to_complex(&mut a, i, o).unwrap());
    // Odd call sizes: n mod 4 and the decimation phase must carry on.
    let mut b =
        RealToComplexState::<C32, _, _>::new(t.clone(), vec![C32::default(); n - 1]).unwrap();
    let mut got = Vec::new();
    let mut buf = [C32::default(); 400];
    let mut pos = 0;
    for len in [1usize, 3, 5, 7, 11, 13, 17, 1, 2, 3].iter().cycle() {
        if pos >= x.len() {
            break;
        }
        let len = (*len).min(x.len() - pos);
        let before = b.n_mod_4();
        let p = Fallback
            .real_to_complex(&mut b, &x[pos..pos + len], &mut buf)
            .unwrap();
        assert_eq!(p.consumed, len);
        assert_eq!(b.n_mod_4(), (before + len) % 4);
        assert_eq!(b.countdown(), b.n_mod_4() & 1);
        got.extend_from_slice(&buf[..p.produced]);
        pos += len;
    }
    assert_eq!(c32_bits(&want), c32_bits(&got));
    // After a reset the stream restarts: n = 0, zero history.
    b.reset();
    assert_eq!(b.n_mod_4(), 0);
    assert!(b.history().iter().all(|s| *s == C32::default()));
    let again = whole(&x[..101], |i, o| {
        Fallback.real_to_complex(&mut b, i, o).unwrap()
    });
    assert_eq!(c32_bits(&want[..51]), c32_bits(&again));
    // An odd count of samples before the reset does not leave a phase behind.
    b.reset();
    let mut buf = [C32::default(); 4];
    Fallback.real_to_complex(&mut b, &x[..3], &mut buf).unwrap();
    assert_eq!(b.n_mod_4(), 3);
    b.reset();
    assert_eq!((b.n_mod_4(), b.countdown()), (0, 0));
}

#[cfg(feature = "reference")]
mod reference {
    use super::*;
    use sdrk_core::Reference;

    fn snr_db(want: &[(f64, f64)], got: &[C32]) -> f64 {
        let (mut sig, mut err) = (0.0, 0.0);
        for (w, g) in want.iter().zip(got) {
            sig += w.0 * w.0 + w.1 * w.1;
            err += (w.0 - f64::from(g.re)).powi(2) + (w.1 - f64::from(g.im)).powi(2);
        }
        10.0 * (sig / err).log10()
    }

    #[test]
    fn fallback_meets_the_contract_against_the_reference() {
        let mut rng = Lcg(11);
        for n in LENS {
            let t = taps_f32(n);
            let x = rand_c32(&mut rng, 3000, 1.0);
            let mut a =
                HalfbandState::<C32, _, _>::new(t.clone(), vec![C32::default(); n - 1]).unwrap();
            let mut b =
                HalfbandState::<C32, _, _>::new(t.clone(), vec![C32::default(); n - 1]).unwrap();
            let y = whole(&x, |i, o| Fallback.halfband_decimate(&mut a, i, o).unwrap());
            let r = whole(&x, |i, o| {
                Reference.halfband_decimate(&mut b, i, o).unwrap()
            });
            let want: Vec<(f64, f64)> = r
                .iter()
                .map(|s| (f64::from(s.re), f64::from(s.im)))
                .collect();
            assert!(snr_db(&want, &y) > 120.0, "n {n}");

            let xr: Vec<f32> = x.iter().map(|s| s.re).collect();
            let mut a =
                RealToComplexState::<C32, _, _>::new(t.clone(), vec![C32::default(); n - 1])
                    .unwrap();
            let mut b =
                RealToComplexState::<C32, _, _>::new(t.clone(), vec![C32::default(); n - 1])
                    .unwrap();
            let y = whole(&xr, |i, o| Fallback.real_to_complex(&mut a, i, o).unwrap());
            let r = whole(&xr, |i, o| Reference.real_to_complex(&mut b, i, o).unwrap());
            let want: Vec<(f64, f64)> = r
                .iter()
                .map(|s| (f64::from(s.re), f64::from(s.im)))
                .collect();
            assert!(snr_db(&want, &y) > 120.0, "r2c n {n}");

            // Q15 is bit-exact on every chunk plan.
            let tq = taps_q15(n);
            let xq = rand_cq15(&mut rng, 2000);
            let mut b =
                HalfbandState::<CQ15, _, _>::new(tq.clone(), vec![CQ15::default(); n - 1]).unwrap();
            let r = whole(&xq, |i, o| {
                Reference.halfband_decimate(&mut b, i, o).unwrap()
            });
            let mut a =
                HalfbandState::<CQ15, _, _>::new(tq.clone(), vec![CQ15::default(); n - 1]).unwrap();
            let y = drive(&xq, 9, |i, o| {
                Fallback.halfband_decimate(&mut a, i, o).unwrap()
            });
            assert_eq!(r, y);
            let xr: Vec<Q15> = xq.iter().map(|s| s.re).collect();
            let mut b =
                HalfbandRealState::<Q15, _, _>::new(tq.clone(), vec![Q15(0); n - 1]).unwrap();
            let r = whole(&xr, |i, o| {
                Reference.halfband_decimate_real(&mut b, i, o).unwrap()
            });
            let mut a =
                HalfbandRealState::<Q15, _, _>::new(tq.clone(), vec![Q15(0); n - 1]).unwrap();
            let y = drive(&xr, 4, |i, o| {
                Fallback.halfband_decimate_real(&mut a, i, o).unwrap()
            });
            assert_eq!(r, y);
            let mut b =
                RealToComplexState::<CQ15, _, _>::new(tq.clone(), vec![CQ15::default(); n - 1])
                    .unwrap();
            let r = whole(&xr, |i, o| Reference.real_to_complex(&mut b, i, o).unwrap());
            let mut a =
                RealToComplexState::<CQ15, _, _>::new(tq, vec![CQ15::default(); n - 1]).unwrap();
            let y = drive(&xr, 6, |i, o| {
                Fallback.real_to_complex(&mut a, i, o).unwrap()
            });
            assert_eq!(r, y);
        }
    }

    #[test]
    fn the_reference_follows_the_output_space_rules() {
        let t = taps_q15(7);
        let mut s = HalfbandRealState::<Q15, _, _>::new(t.clone(), vec![Q15(0); 6]).unwrap();
        let x = [Q15(100); 9];
        let mut y = [Q15(0); 2];
        let p = Reference
            .halfband_decimate_real(&mut s, &x, &mut y)
            .unwrap();
        assert_eq!((p.consumed, p.produced), (4, 2));
        let mut s = RealToComplexState::<CQ15, _, _>::new(t, vec![CQ15::default(); 6]).unwrap();
        let mut yc = [CQ15::default(); 2];
        let p = Reference.real_to_complex(&mut s, &x, &mut yc).unwrap();
        assert_eq!((p.consumed, p.produced), (4, 2));
        assert_eq!(s.n_mod_4(), 0);
    }
}

#[test]
fn extremes_do_not_panic() {
    let n = 23;
    let t = taps_f32(n);
    let specials = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN_POSITIVE,
        1e-40,
    ];
    let x: Vec<f32> = (0..600).map(|i| specials[i % specials.len()]).collect();
    let mut y = vec![C32::default(); 400];
    let mut s =
        RealToComplexState::<C32, _, _>::new(t.clone(), vec![C32::default(); n - 1]).unwrap();
    Fallback.real_to_complex(&mut s, &x, &mut y).unwrap();
    let mut s = HalfbandRealState::<f32, _, _>::new(t.clone(), vec![0.0; n - 1]).unwrap();
    let mut yr = vec![0.0f32; 400];
    Fallback
        .halfband_decimate_real(&mut s, &x, &mut yr)
        .unwrap();
    let xc: Vec<C32> = x
        .iter()
        .zip(x.iter().rev())
        .map(|(&a, &b)| C32::new(a, b))
        .collect();
    let mut s = HalfbandState::<C32, _, _>::new(t, vec![C32::default(); n - 1]).unwrap();
    Fallback.halfband_decimate(&mut s, &xc, &mut y).unwrap();
    // Silence in, silence out.
    let mut s = RealToComplexState::<C32, _, _>::new(taps_f32(7), vec![C32::default(); 6]).unwrap();
    let p = Fallback
        .real_to_complex(&mut s, &[0.0; 100], &mut y)
        .unwrap();
    assert!(y[..p.produced].iter().all(|v| v.re == 0.0 && v.im == 0.0));

    // Q15 corners: all -32768, all 32767, alternating.
    let tq = taps_q15(n);
    for pattern in [
        vec![Q15(-32768); 300],
        vec![Q15(32767); 300],
        (0..300)
            .map(|i| Q15(if i % 2 == 0 { -32768 } else { 32767 }))
            .collect(),
        vec![Q15(0); 300],
    ] {
        let mut s = RealToComplexState::<CQ15, _, _>::new(tq.clone(), vec![CQ15::default(); n - 1])
            .unwrap();
        let mut y = vec![CQ15::default(); 200];
        let p = Fallback.real_to_complex(&mut s, &pattern, &mut y).unwrap();
        assert_eq!(p.produced, 150);
        let mut s = HalfbandRealState::<Q15, _, _>::new(tq.clone(), vec![Q15(0); n - 1]).unwrap();
        let mut y = vec![Q15(0); 200];
        Fallback
            .halfband_decimate_real(&mut s, &pattern, &mut y)
            .unwrap();
    }
    // Taps at the Q15 corners are fine as long as the even offsets are zero.
    let mut tq = vec![Q15(0); 7];
    tq[3] = Q15(-32768);
    tq[2] = Q15(32767);
    tq[4] = Q15(-32768);
    let mut s = HalfbandRealState::<Q15, _, _>::new(tq, vec![Q15(0); 6]).unwrap();
    let mut y = vec![Q15(0); 20];
    Fallback
        .halfband_decimate_real(&mut s, &[Q15(-32768); 40], &mut y)
        .unwrap();
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<HalfbandState<C32, Vec<f32>, Vec<C32>>>();
    assert_send::<HalfbandState<CQ15, [Q15; 7], [CQ15; 6]>>();
    assert_send::<HalfbandRealState<f32, Vec<f32>, Vec<f32>>>();
    assert_send::<HalfbandRealState<Q15, Vec<Q15>, Vec<Q15>>>();
    assert_send::<RealToComplexState<C32, Vec<f32>, Vec<C32>>>();
    assert_send::<RealToComplexState<CQ15, Vec<Q15>, Vec<CQ15>>>();
    assert_sync::<HalfbandState<C32, Vec<f32>, Vec<C32>>>();
    assert_sync::<RealToComplexState<CQ15, Vec<Q15>, Vec<CQ15>>>();
    assert_send::<HalfbandParts<'static, f32, C32>>();
    assert_send::<RealToComplexParts<'static, Q15, CQ15>>();
    assert_send::<Fallback>();
}
