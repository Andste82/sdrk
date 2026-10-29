//! Smoke tests of the references of the fallback round: every function links and returns
//! plausible values. Run through the benchmark workspace with `SDRK_REFS_DIR` set as xtask sets
//! it (`benches/target/refs/<platform>`); a library that is not linked skips its tests.

use cref_sys::cmsis_dsp as cm;
use cref_sys::{
    GrFmDeemph, GrFrequencyModulatorFc, GrQuadratureDemodCf, GrRotator, LiquidDcBlocker,
    LiquidDcBlockerCrcf,
};
use std::f64::consts::PI;

/// Skips a test whose library is not linked.
macro_rules! require {
    ($available:expr, $name:literal) => {
        if !$available {
            eprintln!(concat!("skipped: ", $name, " is not linked"));
            return;
        }
    };
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[track_caller]
fn assert_close(a: f64, b: f64, tol: f64) {
    assert!(close(a, b, tol), "{a} differs from {b} by more than {tol}");
}

/// A deterministic test signal in [-0.9, 0.9].
fn signal(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (f64::from(x) / f64::from(u32::MAX) * 1.8 - 0.9) as f32
        })
        .collect()
}

fn to_q15(x: &[f32]) -> Vec<i16> {
    x.iter()
        .map(|&v| (f64::from(v) * 32768.0).round() as i16)
        .collect()
}

fn pairs<T: Copy>(x: &[T]) -> Vec<[T; 2]> {
    let (chunks, rest) = x.as_chunks::<2>();
    assert!(rest.is_empty(), "odd length");
    chunks.to_vec()
}

#[test]
fn cmsis_conversions() {
    require!(cm::AVAILABLE, "CMSIS-DSP");
    let q7 = [64i8, -128, 127, 0];
    let mut f = [0.0f32; 4];
    cm::q7_to_float(&q7, &mut f);
    assert_eq!(f, [0.5, -1.0, 127.0 / 128.0, 0.0]);
    let mut q15 = [0i16; 4];
    cm::q7_to_q15(&q7, &mut q15);
    assert_eq!(q15, [16384, -32768, 32512, 0]);
    let mut back = [0.0f32; 4];
    cm::q15_to_float(&q15, &mut back);
    assert_eq!(back, [0.5, -1.0, 32512.0 / 32768.0, 0.0]);
    let src = [0.5f32, -0.25, 2.0, -2.0];
    let mut q = [0i16; 4];
    cm::float_to_q15(&src, &mut q);
    assert_eq!(q, [16384, -8192, 32767, -32768]);
    let mut q7b = [0i8; 4];
    cm::float_to_q7(&src, &mut q7b);
    assert_eq!(q7b, [64, -32, 127, -128]);
    cm::q15_to_q7(&[16384, -8192, 255, -1], &mut q7b);
    assert_eq!(q7b, [64, -32, 0, -1]);
}

#[test]
fn cmsis_dot_products() {
    require!(cm::AVAILABLE, "CMSIS-DSP");
    let a = signal(37, 1);
    let b = signal(37, 2);
    let want: f64 = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    assert_close(f64::from(cm::dot_prod_f32(&a, &b)), want, 1e-5);
    let (qa, qb) = (to_q15(&a), to_q15(&b));
    let want_q: i64 = qa
        .iter()
        .zip(&qb)
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum();
    assert_eq!(cm::dot_prod_q15(&qa, &qb), want_q);

    let (ca, cb) = (pairs(&a[..36]), pairs(&b[..36]));
    let mut want_c = [0.0f64; 2];
    for (x, y) in ca.iter().zip(&cb) {
        let (xr, xi, yr, yi) = (
            f64::from(x[0]),
            f64::from(x[1]),
            f64::from(y[0]),
            f64::from(y[1]),
        );
        want_c[0] += xr * yr - xi * yi;
        want_c[1] += xr * yi + xi * yr;
    }
    let got = cm::cmplx_dot_prod_f32(&ca, &cb);
    assert_close(f64::from(got[0]), want_c[0], 1e-5);
    assert_close(f64::from(got[1]), want_c[1], 1e-5);
    let (qca, qcb) = (pairs(&to_q15(&a[..36])), pairs(&to_q15(&b[..36])));
    let got_q = cm::cmplx_dot_prod_q15(&qca, &qcb);
    // Q8.24: the Q30 sums shifted right by 6.
    assert_close(f64::from(got_q[0]) / f64::from(1 << 24), want_c[0], 1e-3);
    assert_close(f64::from(got_q[1]) / f64::from(1 << 24), want_c[1], 1e-3);
}

#[test]
fn cmsis_scale_and_products() {
    require!(cm::AVAILABLE, "CMSIS-DSP");
    let a = signal(21, 3);
    let b = signal(21, 4);
    let mut out = [0.0f32; 21];
    cm::scale_f32(&a, 0.75, &mut out);
    for (o, x) in out.iter().zip(&a) {
        assert_eq!(*o, x * 0.75);
    }
    cm::mult_f32(&a, &b, &mut out);
    for ((o, x), y) in out.iter().zip(&a).zip(&b) {
        assert_eq!(*o, x * y);
    }
    let (qa, qb) = (to_q15(&a), to_q15(&b));
    let mut q = [0i16; 21];
    cm::scale_q15(&qa, 16384, 1, &mut q);
    assert_eq!(q[..], qa[..], "0.5 * 2^1 is the identity");
    cm::mult_q15(&qa, &qb, &mut q);
    for ((o, &x), &y) in q.iter().zip(&qa).zip(&qb) {
        assert_eq!(i32::from(*o), (i32::from(x) * i32::from(y)) >> 15);
    }

    let (ca, cb) = (pairs(&a[..20]), pairs(&b[..20]));
    let mut cout = [[0.0f32; 2]; 10];
    cm::cmplx_mult_real_f32(&ca, &b[..10], &mut cout);
    for ((o, x), y) in cout.iter().zip(&ca).zip(&b[..10]) {
        assert_eq!(*o, [x[0] * y, x[1] * y]);
    }
    cm::cmplx_mult_cmplx_f32(&ca, &cb, &mut cout);
    for ((o, x), y) in cout.iter().zip(&ca).zip(&cb) {
        assert_close(f64::from(o[0]), f64::from(x[0] * y[0] - x[1] * y[1]), 1e-6);
        assert_close(f64::from(o[1]), f64::from(x[0] * y[1] + x[1] * y[0]), 1e-6);
    }
    cm::cmplx_conj_f32(&ca, &mut cout);
    for (o, x) in cout.iter().zip(&ca) {
        assert_eq!(*o, [x[0], -x[1]]);
    }
    let (qca, qcb) = (pairs(&qa[..20]), pairs(&qb[..20]));
    let mut qout = [[0i16; 2]; 10];
    cm::cmplx_mult_real_q15(&qca, &qb[..10], &mut qout);
    for ((o, x), &y) in qout.iter().zip(&qca).zip(&qb[..10]) {
        assert_eq!(i32::from(o[0]), (i32::from(x[0]) * i32::from(y)) >> 15);
    }
    cm::cmplx_mult_cmplx_q15(&qca, &qcb, &mut qout);
    for ((o, x), y) in qout.iter().zip(&ca).zip(&cb) {
        // Q3.13
        let re = f64::from(x[0]) * f64::from(y[0]) - f64::from(x[1]) * f64::from(y[1]);
        assert_close(f64::from(o[0]) / 8192.0, re, 2e-3);
    }
}

#[test]
fn cmsis_magnitudes_and_log() {
    require!(cm::AVAILABLE, "CMSIS-DSP");
    let a = signal(18, 5);
    let c = pairs(&a);
    let mut out = [0.0f32; 9];
    cm::cmplx_mag_squared_f32(&c, &mut out);
    for (o, x) in out.iter().zip(&c) {
        assert_close(f64::from(*o), f64::from(x[0] * x[0] + x[1] * x[1]), 1e-6);
    }
    cm::cmplx_mag_f32(&c, &mut out);
    for (o, x) in out.iter().zip(&c) {
        assert_close(f64::from(*o), f64::from(x[0]).hypot(f64::from(x[1])), 1e-6);
    }
    let qc = pairs(&to_q15(&a));
    let mut q = [0i16; 9];
    cm::cmplx_mag_squared_q15(&qc, &mut q);
    for (o, x) in q.iter().zip(&c) {
        assert_close(
            f64::from(*o) / 8192.0,
            f64::from(x[0] * x[0] + x[1] * x[1]),
            1e-3,
        );
    }
    cm::cmplx_mag_q15(&qc, &mut q);
    for (o, x) in q.iter().zip(&c) {
        assert_close(
            f64::from(*o) / 16384.0,
            f64::from(x[0]).hypot(f64::from(x[1])),
            1e-3,
        );
    }
    let pos = [0.001f32, 0.5, 1.0, 2.0, 10.0, 1000.0, 1e6, 3.0];
    let mut l = [0.0f32; 8];
    cm::vlog_f32(&pos, &mut l);
    for (o, x) in l.iter().zip(&pos) {
        assert_close(f64::from(*o), f64::from(*x).ln(), 1e-3);
    }
}

#[test]
fn cmsis_sine_cosine_atan2() {
    require!(cm::AVAILABLE, "CMSIS-DSP");
    let deg = [0.0f32, 30.0, 90.0, -45.0, 179.0, -170.0];
    let (mut s, mut c) = ([0.0f32; 6], [0.0f32; 6]);
    cm::sin_cos_f32(&deg, &mut s, &mut c);
    for ((sv, cv), d) in s.iter().zip(&c).zip(&deg) {
        let r = f64::from(*d).to_radians();
        assert_close(f64::from(*sv), r.sin(), 1e-4);
        assert_close(f64::from(*cv), r.cos(), 1e-4);
    }
    let theta: Vec<i32> = deg
        .iter()
        .map(|d| (f64::from(*d) / 180.0 * 2_147_483_648.0) as i32)
        .collect();
    let (mut s31, mut c31) = ([0i32; 6], [0i32; 6]);
    cm::sin_cos_q31(&theta, &mut s31, &mut c31);
    for ((sv, cv), d) in s31.iter().zip(&c31).zip(&deg) {
        let r = f64::from(*d).to_radians();
        assert_close(f64::from(*sv) / 2_147_483_648.0, r.sin(), 1e-6);
        assert_close(f64::from(*cv) / 2_147_483_648.0, r.cos(), 1e-6);
    }
    let rad = [0.0f32, 0.5, 1.0, 2.0, 3.0, 6.0];
    let mut y = [0.0f32; 6];
    // A 512-entry table with linear interpolation: errors of about 2e-5.
    cm::sin_f32(&rad, &mut y);
    for (o, x) in y.iter().zip(&rad) {
        assert_close(f64::from(*o), f64::from(*x).sin(), 1e-4);
    }
    cm::cos_f32(&rad, &mut y);
    for (o, x) in y.iter().zip(&rad) {
        assert_close(f64::from(*o), f64::from(*x).cos(), 1e-4);
    }
    let qx = [0i16, 4096, 8192, 16384, 24576, 30000];
    let mut qy = [0i16; 6];
    cm::sin_q15(&qx, &mut qy);
    for (o, x) in qy.iter().zip(&qx) {
        assert_close(
            f64::from(*o) / 32768.0,
            (f64::from(*x) / 32768.0 * 2.0 * PI).sin(),
            1e-3,
        );
    }
    cm::cos_q15(&qx, &mut qy);
    for (o, x) in qy.iter().zip(&qx) {
        assert_close(
            f64::from(*o) / 32768.0,
            (f64::from(*x) / 32768.0 * 2.0 * PI).cos(),
            1e-3,
        );
    }
    let iq = [
        [0.5f32, 0.5],
        [-0.3, 0.2],
        [-0.1, -0.7],
        [0.9, -0.05],
        [0.0, 0.4],
    ];
    let mut ang = [0.0f32; 5];
    cm::atan2_f32(&iq, &mut ang);
    for (o, x) in ang.iter().zip(&iq) {
        assert_close(f64::from(*o), f64::from(x[1]).atan2(f64::from(x[0])), 1e-5);
    }
    let qiq: Vec<[i16; 2]> = iq
        .iter()
        .map(|x| [(x[0] * 32768.0) as i16, (x[1] * 32768.0) as i16])
        .collect();
    let mut qang = [0i16; 5];
    cm::atan2_q15(&qiq, &mut qang);
    for (o, x) in qang.iter().zip(&iq) {
        // Q2.13 radians
        assert_close(
            f64::from(*o) / 8192.0,
            f64::from(x[1]).atan2(f64::from(x[0])),
            2e-3,
        );
    }
}

/// `y[n] = sum h[k] x[n - k]` in f64 with zeros before the start.
fn convolve(h: &[f64], x: &[f64]) -> Vec<f64> {
    (0..x.len())
        .map(|n| {
            h.iter()
                .enumerate()
                .filter(|(k, _)| *k <= n)
                .map(|(k, hk)| hk * x[n - k])
                .sum()
        })
        .collect()
}

#[test]
fn cmsis_fir_filters() {
    require!(cm::AVAILABLE, "CMSIS-DSP");
    let taps = [0.1f32, 0.2, 0.4, 0.2, 0.1, -0.05, 0.03];
    let x = signal(96, 6);
    let want = convolve(
        &taps.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
        &x.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
    );
    let mut fir = cm::CmsisFirF32::new(&taps, 64).expect("arm_fir_init_f32");
    let mut y = [0.0f32; 96];
    fir.run(&x[..48], &mut y[..48]);
    fir.run(&x[48..], &mut y[48..]);
    for (o, w) in y.iter().zip(&want) {
        assert_close(f64::from(*o), *w, 1e-5);
    }
    let (qt, qx) = (to_q15(&taps), to_q15(&x));
    let mut qfir = cm::CmsisFirQ15::new(&qt, 64).expect("arm_fir_init_q15");
    let mut qy = [0i16; 96];
    qfir.run(&qx[..32], &mut qy[..32]);
    qfir.run(&qx[32..], &mut qy[32..]);
    for (o, w) in qy.iter().zip(&want) {
        assert_close(f64::from(*o) / 32768.0, *w, 1e-3);
    }

    // Decimation by 3: CMSIS-DSP's outputs are y_full[3 m], the phase of SDRk's fir_decimate.
    let mut dec = cm::CmsisFirDecimateF32::new(&taps, 3, 48).expect("arm_fir_decimate_init_f32");
    let mut yd = [0.0f32; 32];
    dec.run(&x[..48], &mut yd[..16]);
    dec.run(&x[48..], &mut yd[16..]);
    for (m, o) in yd.iter().enumerate() {
        assert_close(f64::from(*o), want[3 * m], 1e-5);
    }
    let mut qdec = cm::CmsisFirDecimateQ15::new(&qt, 3, 48).expect("arm_fir_decimate_init_q15");
    let mut qyd = [0i16; 32];
    qdec.run(&qx[..48], &mut qyd[..16]);
    qdec.run(&qx[48..], &mut qyd[16..]);
    for (m, o) in qyd.iter().enumerate() {
        assert_close(f64::from(*o) / 32768.0, want[3 * m], 1e-3);
    }

    // Interpolation by 2 with 8 taps: y[j] = sum h[k] u[j - k], u the zero-stuffed input.
    let itaps = [0.05f32, 0.25, 0.5, 0.75, 0.75, 0.5, 0.25, 0.05];
    let u: Vec<f64> = x[..24].iter().flat_map(|&v| [f64::from(v), 0.0]).collect();
    let iwant = convolve(&itaps.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(), &u);
    let mut int =
        cm::CmsisFirInterpolateF32::new(&itaps, 2, 24).expect("arm_fir_interpolate_init_f32");
    let mut yi = [0.0f32; 48];
    int.run(&x[..24], &mut yi);
    for (o, w) in yi.iter().zip(&iwant) {
        assert_close(f64::from(*o), *w, 1e-5);
    }
    let mut qint = cm::CmsisFirInterpolateQ15::new(
        &to_q15(&itaps[..]).iter().map(|v| v / 2).collect::<Vec<_>>(),
        2,
        24,
    )
    .expect("arm_fir_interpolate_init_q15");
    let mut qyi = [0i16; 48];
    qint.run(&qx[..24], &mut qyi);
    for (o, w) in qyi.iter().zip(&iwant) {
        assert_close(f64::from(*o) / 32768.0, w / 2.0, 1e-3);
    }
    assert!(
        cm::CmsisFirInterpolateF32::new(&taps, 2, 24).is_none(),
        "7 taps, factor 2"
    );
}

#[test]
fn cmsis_ffts() {
    require!(cm::AVAILABLE, "CMSIS-DSP");
    for len in [64usize, 1024, 4096] {
        // A complex tone at bin 3 gives N at bin 3 (f32) and about 1 (Q15, scaled by 1/N).
        let tone: Vec<[f64; 2]> = (0..len)
            .map(|n| {
                let p = 2.0 * PI * 3.0 * n as f64 / len as f64;
                [0.5 * p.cos(), 0.5 * p.sin()]
            })
            .collect();
        let mut data: Vec<[f32; 2]> = tone.iter().map(|v| [v[0] as f32, v[1] as f32]).collect();
        let fft = cm::CmsisCfftF32::new(len).expect("cfft f32");
        fft.forward(&mut data);
        let peak = 0.5 * len as f64;
        assert_close(f64::from(data[3][0]), peak, 1e-3 * peak);
        assert!(f64::from(data[5][0]).abs() < 1e-3 * peak);
        let mut q: Vec<[i16; 2]> = tone
            .iter()
            .map(|v| {
                [
                    (v[0] * 32768.0).round() as i16,
                    (v[1] * 32768.0).round() as i16,
                ]
            })
            .collect();
        let qfft = cm::CmsisCfftQ15::new(len).expect("cfft q15");
        qfft.forward(&mut q);
        let scaled = f64::from(q[3][0]) / 32768.0 * len as f64;
        assert_close(scaled, peak, 0.02 * peak);
    }
    assert!(cm::CmsisCfftF32::new(100).is_none());
    let len = 1024;
    let mut input: Vec<f32> = (0..len)
        .map(|n| (2.0 * PI * 5.0 * n as f64 / len as f64).cos() as f32)
        .collect();
    let mut out = vec![0.0f32; len];
    let mut rfft = cm::CmsisRfftFastF32::new(len).expect("rfft fast f32");
    rfft.forward(&mut input, &mut out);
    // Packed: out[2k], out[2k + 1] is bin k for 1 <= k < N/2.
    assert_close(f64::from(out[10]), len as f64 / 2.0, 1e-2);
    assert!(f64::from(out[0]).abs() < 1e-2 && f64::from(out[12]).abs() < 1e-2);
}

#[test]
fn gnuradio_rotator() {
    let Some(mut rot) = GrRotator::new([1.0, 0.0], [0.1f32.cos(), 0.1f32.sin()]) else {
        eprintln!("skipped: the GNU Radio rotator is not linked");
        return;
    };
    let n = 2000;
    let input = vec![[0.5f32, 0.0]; n];
    let mut out = vec![[0.0f32; 2]; n];
    rot.rotate_n(&input, &mut out);
    for (i, o) in out.iter().enumerate() {
        let p = 0.1 * i as f64;
        assert_close(f64::from(o[0]), 0.5 * p.cos(), 1e-4);
        assert_close(f64::from(o[1]), 0.5 * p.sin(), 1e-4);
    }
    assert_eq!(rot.state().counter, 2000);
}

#[test]
fn gnuradio_frequency_modulator() {
    let sensitivity = 0.3f32;
    let Some(mut fm) = GrFrequencyModulatorFc::new(sensitivity) else {
        eprintln!("skipped: the GNU Radio frequency modulator is not linked");
        return;
    };
    let x = signal(500, 7);
    let mut out = [[0.0f32; 2]; 500];
    fm.work(&x, &mut out);
    let mut phase = 0.0f64;
    for (o, v) in out.iter().zip(&x) {
        phase += f64::from(sensitivity) * f64::from(*v);
        assert_close(f64::from(o[0]), phase.cos(), 2e-3);
        assert_close(f64::from(o[1]), phase.sin(), 2e-3);
    }
    // `fmodf` keeps the sign of its first argument, so GNU Radio's phase lies in (-3 pi, pi).
    assert!(fm.phase() < std::f32::consts::PI && fm.phase() > -3.0 * std::f32::consts::PI);
}

#[test]
fn gnuradio_quadrature_demod() {
    let gain = 2.0f32;
    let Some(mut qd) = GrQuadratureDemodCf::new(gain) else {
        eprintln!("skipped: the GNU Radio quadrature demodulator is not linked");
        return;
    };
    let w = 0.2f64;
    let input: Vec<[f32; 2]> = (0..257)
        .map(|n| [(w * n as f64).cos() as f32, (w * n as f64).sin() as f32])
        .collect();
    let mut out = [0.0f32; 256];
    qd.work(&input, &mut out);
    for o in &out {
        assert_close(f64::from(*o), 2.0 * w, 1e-4);
    }
    let a = GrQuadratureDemodCf::fast_atan2f(1.0, -1.0).expect("fast_atan2f");
    assert_close(f64::from(a), 0.75 * PI, 1e-5);
}

#[test]
fn gnuradio_fm_deemph() {
    let Some(mut de) = GrFmDeemph::new(48_000.0, 75e-6) else {
        eprintln!("skipped: the GNU Radio fm_deemph filter is not linked");
        return;
    };
    let [b0, b1, a1] = de.taps();
    assert_close(b0, b1, 1e-15);
    assert_close((b0 + b1) / (1.0 - a1), 1.0, 1e-12);
    let input = vec![0.5f32; 2000];
    let mut out = [0.0f32; 2000];
    de.work(&input[..700], &mut out[..700]);
    de.work(&input[700..], &mut out[700..]);
    assert!(out[0] > 0.0 && out[0] < 0.5);
    assert_close(f64::from(out[1999]), 0.5, 1e-5);
}

#[test]
fn liquid_dc_blockers() {
    let r = 0.99f32;
    let Some(mut dc) = LiquidDcBlocker::new(1.0 - r) else {
        eprintln!("skipped: liquid-dsp's DC blocker is not linked");
        return;
    };
    let x: Vec<f32> = signal(400, 8).iter().map(|v| v + 0.5).collect();
    let mut y = [0.0f32; 400];
    dc.execute_block(&x, &mut y);
    // SDRk's recursion in f64, times liquid's gain sqrt(1 - alpha).
    let g = f64::from(r).sqrt();
    let (mut xp, mut yp) = (0.0f64, 0.0f64);
    for (o, v) in y.iter().zip(&x) {
        let yn = f64::from(*v) - xp + f64::from(1.0 - (1.0 - r)) * yp;
        xp = f64::from(*v);
        yp = yn;
        assert_close(f64::from(*o), g * yn, 1e-4);
    }
    let mut unity = LiquidDcBlocker::new(1.0 - r).expect("dc blocker");
    unity.set_scale(1.0);
    let mut y1 = [0.0f32; 400];
    unity.execute_block(&x, &mut y1);
    assert_close(f64::from(y1[399]), f64::from(y[399]) / g, 1e-4);

    let mut cdc = LiquidDcBlockerCrcf::new(1.0 - r).expect("crcf dc blocker");
    let cx: Vec<[f32; 2]> = x.iter().map(|&v| [v, -v]).collect();
    let mut cy = [[0.0f32; 2]; 400];
    cdc.execute_block(&cx, &mut cy);
    for (c, o) in cy.iter().zip(&y) {
        assert_eq!(c[0], *o);
        assert_eq!(c[1], -*o);
    }
    assert!(LiquidDcBlocker::new(0.0).is_none());
}
