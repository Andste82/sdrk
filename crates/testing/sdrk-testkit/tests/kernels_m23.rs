//! Tests of the M23 kernels that need the testkit: block-size invariance with 20 chunk plans per
//! kernel and type, the frequency mapping of `real_to_complex` (a real tone at `f` appears at
//! `f - fs/4`, its mirror is removed), `halfband_decimate` against `fir_decimate` with the
//! expanded taps, and the extremes through the facade.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::print_stderr,
    clippy::many_single_char_names
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_blocks::design::{halfband_len, halfband_taps, kaiser_beta, to_f32, to_q15};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate};
use sdrk_core::kernels::halfband::{
    HalfbandDecimate, HalfbandDecimateReal, HalfbandRealState, HalfbandState, RealToComplex,
    RealToComplexState,
};
use sdrk_core::{C32, CQ15, Fallback, Q15, Reference, Sample};
use sdrk_testkit::meters::estimate_frequency;
use sdrk_testkit::{ChunkPlan, Pcg32, bit_exact, check_block_invariance};

fn design(n: usize) -> (Vec<f32>, Vec<Q15>) {
    let mut h = vec![0.0f64; n];
    halfband_taps(kaiser_beta(60.0), &mut h).unwrap();
    let mut f = vec![0.0f32; n];
    to_f32(&h, &mut f).unwrap();
    let mut q = vec![Q15(0); n];
    to_q15(&h, &mut q).unwrap();
    (f, q)
}

/// The 20 chunk plans of the invariance tests.
fn plans() -> impl Iterator<Item = ChunkPlan> {
    (0..20u64).map(|seed| ChunkPlan::new(0x2300 + seed, 1 + 37 * seed as usize))
}

fn noise_c32(rng: &mut Pcg32, n: usize) -> Vec<C32> {
    (0..n)
        .map(|_| C32::new((rng.normal() * 0.3) as f32, (rng.normal() * 0.3) as f32))
        .collect()
}

fn noise_f32(rng: &mut Pcg32, n: usize) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * 0.3) as f32).collect()
}

fn q(v: f64) -> Q15 {
    Q15((v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
}

#[test]
fn the_three_kernels_are_block_size_invariant_with_20_plans() {
    let mut rng = Pcg32::new(2300, 1);
    for n in [23usize, 47, 63] {
        let (hf, hq) = design(n);
        let xc = noise_c32(&mut rng, 3000);
        let xr = noise_f32(&mut rng, 3000);
        let xq: Vec<Q15> = xr.iter().map(|&v| q(f64::from(v))).collect();
        let xcq: Vec<CQ15> = xc
            .iter()
            .map(|s| CQ15 {
                re: q(f64::from(s.re)),
                im: q(f64::from(s.im)),
            })
            .collect();
        for plan in plans() {
            let mut a = vec![C32::ZERO; 1600];
            let mut b = vec![C32::ZERO; 1600];
            check_block_invariance(
                &xc,
                &mut a,
                &mut b,
                plan,
                || HalfbandState::<C32, _, _>::new(hf.clone(), vec![C32::ZERO; n - 1]).unwrap(),
                |s, i, o| Sdrk.halfband_decimate(s, i, o),
            )
            .unwrap();
            let mut a = vec![CQ15::ZERO; 1600];
            let mut b = vec![CQ15::ZERO; 1600];
            check_block_invariance(
                &xcq,
                &mut a,
                &mut b,
                plan,
                || HalfbandState::<CQ15, _, _>::new(hq.clone(), vec![CQ15::ZERO; n - 1]).unwrap(),
                |s, i, o| Sdrk.halfband_decimate(s, i, o),
            )
            .unwrap();
            let mut a = vec![0.0f32; 1600];
            let mut b = vec![0.0f32; 1600];
            check_block_invariance(
                &xr,
                &mut a,
                &mut b,
                plan,
                || HalfbandRealState::<f32, _, _>::new(hf.clone(), vec![0.0; n - 1]).unwrap(),
                |s, i, o| Sdrk.halfband_decimate_real(s, i, o),
            )
            .unwrap();
            let mut a = vec![Q15(0); 1600];
            let mut b = vec![Q15(0); 1600];
            check_block_invariance(
                &xq,
                &mut a,
                &mut b,
                plan,
                || HalfbandRealState::<Q15, _, _>::new(hq.clone(), vec![Q15(0); n - 1]).unwrap(),
                |s, i, o| Sdrk.halfband_decimate_real(s, i, o),
            )
            .unwrap();
            let mut a = vec![C32::ZERO; 1600];
            let mut b = vec![C32::ZERO; 1600];
            check_block_invariance(
                &xr,
                &mut a,
                &mut b,
                plan,
                || {
                    RealToComplexState::<C32, _, _>::new(hf.clone(), vec![C32::ZERO; n - 1])
                        .unwrap()
                },
                |s, i, o| Sdrk.real_to_complex(s, i, o),
            )
            .unwrap();
            let mut a = vec![CQ15::ZERO; 1600];
            let mut b = vec![CQ15::ZERO; 1600];
            check_block_invariance(
                &xq,
                &mut a,
                &mut b,
                plan,
                || {
                    RealToComplexState::<CQ15, _, _>::new(hq.clone(), vec![CQ15::ZERO; n - 1])
                        .unwrap()
                },
                |s, i, o| Sdrk.real_to_complex(s, i, o),
            )
            .unwrap();
        }
    }
}

/// `sum_n y[n] e^{-j 2 pi cycles n}` over `y`, normalized by the length.
fn dft(y: &[Complex64], cycles: f64) -> Complex64 {
    let mut acc = Complex64::new(0.0, 0.0);
    for (n, v) in y.iter().enumerate() {
        let a = -2.0 * std::f64::consts::PI * cycles * n as f64;
        acc += v * Complex64::new(a.cos(), a.sin());
    }
    acc / y.len() as f64
}

fn mag(c: Complex64) -> f64 {
    c.re.hypot(c.im)
}

fn to_c64(y: &[C32]) -> Vec<Complex64> {
    y.iter()
        .map(|s| Complex64::new(f64::from(s.re), f64::from(s.im)))
        .collect()
}

fn to_c64_q(y: &[CQ15]) -> Vec<Complex64> {
    y.iter()
        .map(|s| Complex64::new(f64::from(s.re.0) / 32768.0, f64::from(s.im.0) / 32768.0))
        .collect()
}

/// The transition width and attenuation of the design used below.
const TRANSITION: f64 = 0.1;
const ATTENUATION: f64 = 60.0;

fn tone(f: f64, amp: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| amp * (2.0 * std::f64::consts::PI * f * i as f64 + 0.3).cos())
        .collect()
}

#[test]
fn a_real_tone_appears_at_f_minus_fs_over_4_and_the_mirror_is_removed() {
    let n = halfband_len(TRANSITION, ATTENUATION, 1.0).unwrap();
    assert_eq!(n % 4, 3);
    let (hf, hq) = design(n);
    // fs = 1 MHz: 1e-6 fs = 1 Hz. The wanted output frequencies f - fs/4 lie inside the pass band
    // (up to 0.2 fs); the tone frequencies are chosen so that the wanted component and its mirror
    // are DFT bins of the steady-state window (output rate fs/2, 3000 outputs).
    let fs = 1.0e6;
    for wanted_cycles in [-0.35f64, -0.2, -0.1, 0.06, 0.1, 0.3, 0.38] {
        // `wanted_cycles` is the output frequency in cycles per output sample (rate fs/2);
        // the input frequency is fs/4 + wanted_cycles * fs / 2.
        let f_out_hz = wanted_cycles * fs / 2.0;
        if f_out_hz.abs() > 0.2 * fs {
            continue;
        }
        let f_in = 0.25 + wanted_cycles / 2.0;
        let x = tone(f_in, 0.5, 8000);
        let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let mut s =
            RealToComplexState::<C32, _, _>::new(hf.clone(), vec![C32::ZERO; n - 1]).unwrap();
        let mut y = vec![C32::ZERO; 4000];
        let p = Sdrk.real_to_complex(&mut s, &xf, &mut y).unwrap();
        assert_eq!((p.consumed, p.produced), (8000, 4000));
        let y = to_c64(&y);
        let steady = &y[500..3500];
        let f_est = estimate_frequency(steady, fs / 2.0);
        assert!(
            (f_est - f_out_hz).abs() <= 1e-6 * fs,
            "tone at {f_in} fs: estimated {f_est} Hz, expected {f_out_hz} Hz"
        );
        // Wanted component amplitude 0.25 (half of the real amplitude); the mirror lands at the
        // negative frequency.
        let wanted = dft(steady, wanted_cycles);
        let mirror = dft(steady, -wanted_cycles);
        assert!((mag(wanted) - 0.25).abs() < 0.01, "{wanted}");
        let rejection = 20.0 * (mag(wanted) / mag(mirror)).log10();
        eprintln!("f_in {f_in:.4} fs: mirror {rejection:.1} dB below");
        assert!(
            rejection >= ATTENUATION - 3.0,
            "f_in {f_in}: the mirror is only {rejection} dB down"
        );

        // The same in CQ15.
        let xq: Vec<Q15> = x.iter().map(|&v| q(v)).collect();
        let mut s =
            RealToComplexState::<CQ15, _, _>::new(hq.clone(), vec![CQ15::ZERO; n - 1]).unwrap();
        let mut y = vec![CQ15::ZERO; 4000];
        Sdrk.real_to_complex(&mut s, &xq, &mut y).unwrap();
        let y = to_c64_q(&y);
        let steady = &y[500..3500];
        assert!(
            (estimate_frequency(steady, fs / 2.0) - f_out_hz).abs() <= 1e-6 * fs,
            "cq15 tone at {f_in} fs"
        );
        let wanted = dft(steady, wanted_cycles);
        let mirror = dft(steady, -wanted_cycles);
        let rejection = 20.0 * (mag(wanted) / mag(mirror)).log10();
        assert!(
            rejection >= ATTENUATION - 3.0,
            "cq15 f_in {f_in}: {rejection} dB"
        );
    }
}

#[test]
fn halfband_decimate_equals_fir_decimate_with_the_expanded_taps() {
    let mut rng = Pcg32::new(2301, 1);
    for n in [11usize, 23, 63] {
        let (hf, hq) = design(n);
        let x = noise_c32(&mut rng, 4001);
        // The fallback sums the even taps and then the centre tap (D88): the same as two dense
        // filters, the even taps alone and the centre tap alone, added in `f32`.
        let m = (n - 1) / 2;
        let mut even = hf.clone();
        even[m] = 0.0;
        let mut centre = vec![0.0f32; n];
        centre[m] = hf[m];
        let mut a = FirDecimState::<C32, _, _>::new(even, vec![C32::ZERO; n - 1], 2).unwrap();
        let mut c = FirDecimState::<C32, _, _>::new(centre, vec![C32::ZERO; n - 1], 2).unwrap();
        let mut b = HalfbandState::<C32, _, _>::new(hf, vec![C32::ZERO; n - 1]).unwrap();
        let (mut ya, mut yc, mut yb) = (
            vec![C32::ZERO; 2001],
            vec![C32::ZERO; 2001],
            vec![C32::ZERO; 2001],
        );
        let pa = Fallback.fir_decimate(&mut a, &x, &mut ya).unwrap();
        Fallback.fir_decimate(&mut c, &x, &mut yc).unwrap();
        let pb = Fallback.halfband_decimate(&mut b, &x, &mut yb).unwrap();
        assert_eq!(pa, pb);
        for (a, c) in ya.iter_mut().zip(&yc) {
            *a = C32::new(a.re + c.re, a.im + c.im);
        }
        assert_eq!(ya, yb);
        let xq: Vec<CQ15> = x
            .iter()
            .map(|s| CQ15 {
                re: q(f64::from(s.re) * 3.0),
                im: q(f64::from(s.im) * 3.0),
            })
            .collect();
        let mut a =
            FirDecimState::<CQ15, _, _>::new(hq.clone(), vec![CQ15::ZERO; n - 1], 2).unwrap();
        let mut b = HalfbandState::<CQ15, _, _>::new(hq, vec![CQ15::ZERO; n - 1]).unwrap();
        let (mut ya, mut yb) = (vec![CQ15::ZERO; 2001], vec![CQ15::ZERO; 2001]);
        Reference.fir_decimate(&mut a, &xq, &mut ya).unwrap();
        Sdrk.halfband_decimate(&mut b, &xq, &mut yb).unwrap();
        assert_eq!(bit_exact(&ya, &yb), None);
    }
}

#[test]
fn extremes_through_the_facade() {
    let n = 23;
    let (hf, hq) = design(n);
    let specials = [
        0.0f32,
        1.0,
        -1.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN_POSITIVE,
    ];
    let x: Vec<f32> = (0..1000).map(|i| specials[i % specials.len()]).collect();
    let mut y = vec![C32::ZERO; 600];
    let mut s = RealToComplexState::<C32, _, _>::new(hf.clone(), vec![C32::ZERO; n - 1]).unwrap();
    Sdrk.real_to_complex(&mut s, &x, &mut y).unwrap();
    let mut yr = vec![0.0f32; 600];
    let mut s = HalfbandRealState::<f32, _, _>::new(hf.clone(), vec![0.0; n - 1]).unwrap();
    Sdrk.halfband_decimate_real(&mut s, &x, &mut yr).unwrap();
    let xc: Vec<C32> = x
        .iter()
        .zip(x.iter().rev())
        .map(|(&a, &b)| C32::new(a, b))
        .collect();
    let mut s = HalfbandState::<C32, _, _>::new(hf, vec![C32::ZERO; n - 1]).unwrap();
    Sdrk.halfband_decimate(&mut s, &xc, &mut y).unwrap();
    // Q15 corners: full scale, alternating signs, silence.
    for pattern in [
        vec![Q15(-32768); 700],
        vec![Q15(32767); 700],
        (0..700)
            .map(|i| Q15(if i % 3 == 0 { -32768 } else { 32767 }))
            .collect::<Vec<_>>(),
        vec![Q15(0); 700],
    ] {
        let mut s =
            RealToComplexState::<CQ15, _, _>::new(hq.clone(), vec![CQ15::ZERO; n - 1]).unwrap();
        let mut y = vec![CQ15::ZERO; 400];
        Sdrk.real_to_complex(&mut s, &pattern, &mut y).unwrap();
        let mut s = HalfbandRealState::<Q15, _, _>::new(hq.clone(), vec![Q15(0); n - 1]).unwrap();
        let mut yr = vec![Q15(0); 400];
        Sdrk.halfband_decimate_real(&mut s, &pattern, &mut yr)
            .unwrap();
        // Silence in, silence out.
        if pattern.iter().all(|v| v.0 == 0) {
            assert!(y.iter().all(|v| *v == CQ15::ZERO) && yr.iter().all(|v| v.0 == 0));
        }
    }
}
