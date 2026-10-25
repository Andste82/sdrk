//! Integration tests of the facade `Sdrk`.

#![allow(
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    clippy::items_after_statements,
    clippy::too_many_lines
)]
use sdrk::kernels::basic::Scale;
use sdrk::{C32, CQ15, Error, Fallback, Q15, Result, SampleType, Sdrk, backend_for};

/// The name `backend_for` reports for the accelerated kernels.
const ACCEL: &str = if sdrk::SIMD_BACKEND {
    "simd"
} else {
    "fallback"
};

/// The two results agree to within `1e-5` of the largest magnitude (float kernels whose
/// accelerated variant forms its sums in a different order).
fn assert_close(want: &[f32], got: &[f32]) {
    let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        assert!((w - g).abs() <= 1e-5 * scale, "i={i} want={w} got={g}");
    }
    assert_eq!(want.len(), got.len());
}

fn flat(x: &[C32]) -> Vec<f32> {
    x.iter().flat_map(|v| [v.re, v.im]).collect()
}

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

#[test]
fn sdrk_matches_fallback() -> Result<()> {
    let mut s = 99u64;
    let c32: Vec<C32> = (0..1024)
        .map(|_| {
            C32::new(
                lcg(&mut s) as f32 / 1e9 - 2.0,
                lcg(&mut s) as f32 / 1e9 - 2.0,
            )
        })
        .collect();
    let (mut a, mut b) = (vec![C32::default(); 1024], vec![C32::default(); 1024]);
    Sdrk.scale(&c32, &mut a, 0.7)?;
    Fallback.scale(&c32, &mut b, 0.7)?;
    assert_eq!(a, b);

    let q: Vec<CQ15> = (0..1024)
        .map(|_| CQ15 {
            re: Q15(lcg(&mut s) as i16),
            im: Q15(lcg(&mut s) as i16),
        })
        .collect();
    let (mut a, mut b) = (vec![CQ15::default(); 1024], vec![CQ15::default(); 1024]);
    Sdrk.scale(&q, &mut a, Q15(-9999))?;
    Fallback.scale(&q, &mut b, Q15(-9999))?;
    assert_eq!(a, b);
    Ok(())
}

#[test]
fn facade_reports_length_mismatch() {
    let mut out = [C32::default(); 1];
    assert_eq!(
        Sdrk.scale(&[C32::default(); 2], &mut out, 1.0),
        Err(Error::LengthMismatch {
            expected: 2,
            got: 1
        })
    );
}

#[test]
fn backend_table() {
    assert_eq!(backend_for("scale", SampleType::C32), "fallback");
    assert_eq!(backend_for("scale", SampleType::CQ15), "fallback");
    assert_eq!(backend_for("scale", SampleType::F32), "none");
    assert_eq!(backend_for("scale", SampleType::Q15), "none");
    assert_eq!(backend_for("unknown", SampleType::C32), "none");
}

#[test]
fn sdrk_is_copy_default_send_sync() {
    fn assert_all<T: Copy + Default + Send + Sync>() {}
    assert_all::<Sdrk>();
}

#[test]
fn backend_table_lists_the_m05_kernels() {
    for (kernel, ty) in [
        ("convert", SampleType::C32),
        ("convert", SampleType::CQ15),
        ("rotate", SampleType::C32),
        ("rotate", SampleType::CQ15),
        ("fir", SampleType::C32),
        ("fir", SampleType::CQ15),
    ] {
        assert_eq!(backend_for(kernel, ty), ACCEL, "{kernel} {ty:?}");
    }
    for (kernel, ty) in [
        ("convert_real", SampleType::F32),
        ("convert_real", SampleType::Q15),
    ] {
        assert_eq!(backend_for(kernel, ty), "fallback", "{kernel} {ty:?}");
    }
    assert_eq!(backend_for("convert", SampleType::F32), "none");
    assert_eq!(backend_for("convert_real", SampleType::C32), "none");
    assert_eq!(backend_for("rotate", SampleType::Q15), "none");
    assert_eq!(backend_for("fir", SampleType::F32), "none");
}

#[test]
fn m05_kernels_agree_with_the_fallback() -> Result<()> {
    use sdrk::kernels::convert::{Convert, ConvertReal};
    use sdrk::kernels::filter::{Fir, FirState};
    use sdrk::kernels::mix::{Nco, Rotate};
    use sdrk_device::formats;

    let mut s = 4u64;
    let bytes: Vec<u8> = (0..256).map(|_| (lcg(&mut s) >> 24) as u8).collect();
    let (mut a, mut b) = (vec![C32::default(); 64], vec![C32::default(); 64]);
    assert_eq!(Sdrk.convert(&formats::CS16, &bytes, &mut a)?, 64);
    Fallback.convert(&formats::CS16, &bytes, &mut b)?;
    assert_eq!(a, b);
    let (mut a, mut b) = (vec![CQ15::default(); 64], vec![CQ15::default(); 64]);
    Sdrk.convert(&formats::CU8, &bytes[..128], &mut a)?;
    Fallback.convert(&formats::CU8, &bytes[..128], &mut b)?;
    assert_eq!(a, b);
    let (mut a, mut b) = ([0f32; 64], [0f32; 64]);
    Sdrk.convert_real(&formats::RS16, &bytes[..128], &mut a)?;
    Fallback.convert_real(&formats::RS16, &bytes[..128], &mut b)?;
    assert_eq!(a, b);
    let (mut a, mut b) = ([Q15(0); 64], [Q15(0); 64]);
    Sdrk.convert_real(&formats::RS16, &bytes[..128], &mut a)?;
    Fallback.convert_real(&formats::RS16, &bytes[..128], &mut b)?;
    assert_eq!(a, b);

    let c32: Vec<C32> = (0..200)
        .map(|_| C32::new(lcg(&mut s) as f32 / 4e9, lcg(&mut s) as f32 / 4e9))
        .collect();
    let (mut a, mut b) = (vec![C32::default(); 200], vec![C32::default(); 200]);
    let (mut na, mut nb) = (Nco::from_increment(1234567), Nco::from_increment(1234567));
    Sdrk.rotate(&mut na, &c32, &mut a)?;
    Fallback.rotate(&mut nb, &c32, &mut b)?;
    assert_close(&flat(&b), &flat(&a));
    assert_eq!(na, nb);

    let cq: Vec<CQ15> = (0..200)
        .map(|_| CQ15 {
            re: Q15(lcg(&mut s) as i16),
            im: Q15(lcg(&mut s) as i16),
        })
        .collect();
    let (mut a, mut b) = (vec![CQ15::default(); 200], vec![CQ15::default(); 200]);
    let (mut na, mut nb) = (Nco::from_increment(7654321), Nco::from_increment(7654321));
    Sdrk.rotate(&mut na, &cq, &mut a)?;
    Fallback.rotate(&mut nb, &cq, &mut b)?;
    assert_eq!((a, na), (b, nb));

    let taps = [0.1f32, 0.2, 0.4, 0.2, 0.1];
    let (mut sa, mut sb) = (
        FirState::<C32, _, _>::new(taps, [C32::default(); 4])?,
        FirState::<C32, _, _>::new(taps, [C32::default(); 4])?,
    );
    let (mut a, mut b) = (vec![C32::default(); 200], vec![C32::default(); 200]);
    Sdrk.fir(&mut sa, &c32, &mut a)?;
    Fallback.fir(&mut sb, &c32, &mut b)?;
    assert_close(&flat(&b), &flat(&a));
    assert_eq!(sa.history(), sb.history());

    let taps = [Q15(1000), Q15(-2000), Q15(30000)];
    let (mut sa, mut sb) = (
        FirState::<CQ15, _, _>::new(taps, [CQ15::default(); 2])?,
        FirState::<CQ15, _, _>::new(taps, [CQ15::default(); 2])?,
    );
    let (mut a, mut b) = (vec![CQ15::default(); 200], vec![CQ15::default(); 200]);
    Sdrk.fir(&mut sa, &cq, &mut a)?;
    Fallback.fir(&mut sb, &cq, &mut b)?;
    assert_eq!((a, sa.history()), (b, sb.history()));
    Ok(())
}

#[test]
fn backend_table_lists_the_m06_kernels() {
    for (kernel, ty) in [
        ("fir_decimate", SampleType::C32),
        ("fir_decimate", SampleType::CQ15),
        ("resample", SampleType::C32),
        ("resample", SampleType::CQ15),
        ("quad_demod", SampleType::C32),
        ("quad_demod", SampleType::CQ15),
        ("mag_sq", SampleType::C32),
        ("mag_sq", SampleType::CQ15),
        ("mag", SampleType::C32),
        ("mag", SampleType::CQ15),
    ] {
        assert_eq!(backend_for(kernel, ty), ACCEL, "{kernel} {ty:?}");
    }
    for (kernel, ty) in [
        ("resample_real", SampleType::F32),
        ("resample_real", SampleType::Q15),
        ("deemphasis", SampleType::F32),
        ("deemphasis", SampleType::Q15),
        ("dc_block", SampleType::C32),
        ("dc_block", SampleType::CQ15),
        ("squelch", SampleType::C32),
        ("squelch", SampleType::CQ15),
        ("dc_block_real", SampleType::F32),
        ("dc_block_real", SampleType::Q15),
        ("agc", SampleType::C32),
        ("agc", SampleType::CQ15),
        ("window", SampleType::C32),
        ("window", SampleType::CQ15),
        ("power_db", SampleType::F32),
        ("power_db", SampleType::Q15),
    ] {
        assert_eq!(backend_for(kernel, ty), "fallback", "{kernel} {ty:?}");
    }
    assert_eq!(backend_for("fir_decimate", SampleType::F32), "none");
    assert_eq!(backend_for("deemphasis", SampleType::C32), "none");
    assert_eq!(backend_for("squelch", SampleType::F32), "none");
    assert_eq!(backend_for("agc", SampleType::F32), "none");
    assert_eq!(backend_for("dc_block_real", SampleType::C32), "none");
    assert_eq!(backend_for("resample_real", SampleType::C32), "none");
    assert_eq!(backend_for("power_db", SampleType::CQ15), "none");
}

#[test]
fn m06_kernels_agree_with_the_fallback() -> Result<()> {
    use sdrk::Rate;
    use sdrk::kernels::decim::{
        FirDecimState, FirDecimate, Resample, ResampleReal, ResampleRealState, ResampleState,
    };
    use sdrk::kernels::demod::{
        DcBlock, DcBlockState, Deemphasis, DeemphasisState, QuadDemod, QuadDemodState,
    };
    use sdrk::kernels::spectrum::{MagSq, PowerDb, Window};

    let mut s = 8u64;
    let c32: Vec<C32> = (0..200)
        .map(|_| {
            C32::new(
                lcg(&mut s) as f32 / 4e9 - 0.5,
                lcg(&mut s) as f32 / 4e9 - 0.5,
            )
        })
        .collect();
    let cq: Vec<CQ15> = (0..200)
        .map(|_| CQ15 {
            re: Q15(lcg(&mut s) as i16),
            im: Q15(lcg(&mut s) as i16),
        })
        .collect();
    let taps = [0.1f32, 0.2, 0.4, 0.2, 0.1];
    let tapsq = [Q15(1000), Q15(-2000), Q15(30000)];

    let (mut sa, mut sb) = (
        FirDecimState::<C32, _, _>::new(taps, [C32::default(); 4], 3)?,
        FirDecimState::<C32, _, _>::new(taps, [C32::default(); 4], 3)?,
    );
    let (mut a, mut b) = (vec![C32::default(); 200], vec![C32::default(); 200]);
    let (pa, pb) = (
        Sdrk.fir_decimate(&mut sa, &c32, &mut a)?,
        Fallback.fir_decimate(&mut sb, &c32, &mut b)?,
    );
    assert_eq!(pa, pb);
    assert_close(&flat(&b[..pb.produced]), &flat(&a[..pa.produced]));
    assert_eq!(sa.history(), sb.history());
    let (mut sa, mut sb) = (
        FirDecimState::<CQ15, _, _>::new(tapsq, [CQ15::default(); 2], 4)?,
        FirDecimState::<CQ15, _, _>::new(tapsq, [CQ15::default(); 2], 4)?,
    );
    let (mut a, mut b) = (vec![CQ15::default(); 200], vec![CQ15::default(); 200]);
    let (pa, pb) = (
        Sdrk.fir_decimate(&mut sa, &cq, &mut a)?,
        Fallback.fir_decimate(&mut sb, &cq, &mut b)?,
    );
    assert_eq!((pa, a), (pb, b));

    let (mut sa, mut sb) = (
        ResampleState::<C32, _, _>::new(taps, [C32::default(); 2], 3, 2)?,
        ResampleState::<C32, _, _>::new(taps, [C32::default(); 2], 3, 2)?,
    );
    let (mut a, mut b) = (vec![C32::default(); 400], vec![C32::default(); 400]);
    let (pa, pb) = (
        Sdrk.resample(&mut sa, &c32, &mut a)?,
        Fallback.resample(&mut sb, &c32, &mut b)?,
    );
    assert_eq!(pa, pb);
    assert_close(&flat(&b[..pb.produced]), &flat(&a[..pa.produced]));
    assert_eq!(sa.phase(), sb.phase());
    let (mut sa, mut sb) = (
        ResampleState::<CQ15, _, _>::new(tapsq, [CQ15::default(); 1], 3, 2)?,
        ResampleState::<CQ15, _, _>::new(tapsq, [CQ15::default(); 1], 3, 2)?,
    );
    let (mut a, mut b) = (vec![CQ15::default(); 400], vec![CQ15::default(); 400]);
    let (pa, pb) = (
        Sdrk.resample(&mut sa, &cq, &mut a)?,
        Fallback.resample(&mut sb, &cq, &mut b)?,
    );
    assert_eq!((pa, a), (pb, b));
    let xr: Vec<f32> = c32.iter().map(|v| v.re).collect();
    let (mut sa, mut sb) = (
        ResampleRealState::<f32, _, _>::new(taps, [0f32; 5], 1, 2)?,
        ResampleRealState::<f32, _, _>::new(taps, [0f32; 5], 1, 2)?,
    );
    let (mut a, mut b) = (vec![0f32; 200], vec![0f32; 200]);
    let (pa, pb) = (
        Sdrk.resample_real(&mut sa, &xr, &mut a)?,
        Fallback.resample_real(&mut sb, &xr, &mut b)?,
    );
    assert_eq!((pa, a), (pb, b));
    let xq: Vec<Q15> = cq.iter().map(|v| v.re).collect();
    let (mut sa, mut sb) = (
        ResampleRealState::<Q15, _, _>::new(tapsq, [Q15(0); 3], 1, 2)?,
        ResampleRealState::<Q15, _, _>::new(tapsq, [Q15(0); 3], 1, 2)?,
    );
    let (mut a, mut b) = (vec![Q15(0); 200], vec![Q15(0); 200]);
    let (pa, pb) = (
        Sdrk.resample_real(&mut sa, &xq, &mut a)?,
        Fallback.resample_real(&mut sb, &xq, &mut b)?,
    );
    assert_eq!((pa, a), (pb, b));

    let (mut sa, mut sb) = (QuadDemodState::<C32>::new(), QuadDemodState::<C32>::new());
    let (mut a, mut b) = (vec![0f32; 200], vec![0f32; 200]);
    Sdrk.quad_demod(&mut sa, &c32, &mut a, 0.7)?;
    Fallback.quad_demod(&mut sb, &c32, &mut b, 0.7)?;
    assert_close(&b, &a);
    assert_eq!(sa, sb);
    let (mut sa, mut sb) = (QuadDemodState::<CQ15>::new(), QuadDemodState::<CQ15>::new());
    let (mut a, mut b) = (vec![Q15(0); 200], vec![Q15(0); 200]);
    Sdrk.quad_demod(&mut sa, &cq, &mut a, Q15(20000))?;
    Fallback.quad_demod(&mut sb, &cq, &mut b, Q15(20000))?;
    for (w, g) in b.iter().zip(&a) {
        assert!((i32::from(w.0) - i32::from(g.0)).abs() <= 2);
    }
    assert_eq!(sa, sb);

    let rate = Rate::hz(48_000)?;
    let (mut sa, mut sb) = (
        DeemphasisState::<f32>::new(50e-6, rate)?,
        DeemphasisState::<f32>::new(50e-6, rate)?,
    );
    let (mut a, mut b) = (vec![0f32; 200], vec![0f32; 200]);
    Sdrk.deemphasis(&mut sa, &xr, &mut a)?;
    Fallback.deemphasis(&mut sb, &xr, &mut b)?;
    assert_eq!((a, sa), (b, sb));
    let (mut sa, mut sb) = (
        DeemphasisState::<Q15>::new(50e-6, rate)?,
        DeemphasisState::<Q15>::new(50e-6, rate)?,
    );
    let (mut a, mut b) = (vec![Q15(0); 200], vec![Q15(0); 200]);
    Sdrk.deemphasis(&mut sa, &xq, &mut a)?;
    Fallback.deemphasis(&mut sb, &xq, &mut b)?;
    assert_eq!((a, sa), (b, sb));

    let (mut sa, mut sb) = (
        DcBlockState::<C32>::new(0.99)?,
        DcBlockState::<C32>::new(0.99)?,
    );
    let (mut a, mut b) = (vec![C32::default(); 200], vec![C32::default(); 200]);
    Sdrk.dc_block(&mut sa, &c32, &mut a)?;
    Fallback.dc_block(&mut sb, &c32, &mut b)?;
    assert_eq!((a, sa), (b, sb));
    let (mut sa, mut sb) = (
        DcBlockState::<CQ15>::new(Q15(32000))?,
        DcBlockState::<CQ15>::new(Q15(32000))?,
    );
    let (mut a, mut b) = (vec![CQ15::default(); 200], vec![CQ15::default(); 200]);
    Sdrk.dc_block(&mut sa, &cq, &mut a)?;
    Fallback.dc_block(&mut sb, &cq, &mut b)?;
    assert_eq!((a, sa), (b, sb));

    let (mut a, mut b) = (vec![C32::default(); 200], vec![C32::default(); 200]);
    Sdrk.window(&c32, &mut a, &xr)?;
    Fallback.window(&c32, &mut b, &xr)?;
    assert_eq!(a, b);
    let (mut a, mut b) = (vec![CQ15::default(); 200], vec![CQ15::default(); 200]);
    Sdrk.window(&cq, &mut a, &xq)?;
    Fallback.window(&cq, &mut b, &xq)?;
    assert_eq!(a, b);
    let (mut a, mut b) = (vec![0f32; 200], vec![0f32; 200]);
    Sdrk.mag_sq(&c32, &mut a)?;
    Fallback.mag_sq(&c32, &mut b)?;
    // No multiply-add, so bit-exact also in the relaxed WASM build.
    assert_eq!(a, b);
    let (mut a, mut b) = (vec![Q15(0); 200], vec![Q15(0); 200]);
    Sdrk.mag_sq(&cq, &mut a)?;
    Fallback.mag_sq(&cq, &mut b)?;
    assert_eq!(a, b);
    let p: Vec<f32> = xr.iter().map(|v| v * v).collect();
    let (mut a, mut b) = (vec![0f32; 200], vec![0f32; 200]);
    Sdrk.power_db(&p, &mut a, 3.0)?;
    Fallback.power_db(&p, &mut b, 3.0)?;
    assert_eq!(a, b);
    Sdrk.power_db(&xq, &mut a, 3.0)?;
    Fallback.power_db(&xq, &mut b, 3.0)?;
    assert_eq!(a, b);
    Ok(())
}

#[test]
fn m09_dot_kernels_delegate_to_the_fallback() -> Result<()> {
    use sdrk::kernels::dot::{Dot, DotReal};
    for (kernel, ty) in [
        ("dot", SampleType::C32),
        ("dot", SampleType::CQ15),
        ("dot_real", SampleType::F32),
        ("dot_real", SampleType::Q15),
    ] {
        assert_eq!(backend_for(kernel, ty), "fallback", "{kernel} {ty:?}");
    }
    assert_eq!(backend_for("dot", SampleType::F32), "none");
    assert_eq!(backend_for("dot_real", SampleType::C32), "none");
    let x = [C32::new(1.0, 2.0), C32::new(3.0, -1.0)];
    let taps = [0.5f32, 2.0];
    assert_eq!(Sdrk.dot(&x, &taps)?, Fallback.dot(&x, &taps)?);
    let q = [Q15(16384), Q15(-8192)];
    assert_eq!(Sdrk.dot_real(&q, &q)?, Fallback.dot_real(&q, &q)?);
    let xq = [CQ15 {
        re: Q15(1000),
        im: Q15(-2000),
    }; 2];
    assert_eq!(Sdrk.dot(&xq, &q)?, Fallback.dot(&xq, &q)?);
    assert_eq!(Sdrk.dot_real(&[1.0f32, 2.0], &[3.0, 4.0])?, 11.0);
    assert!(Sdrk.dot(&x, &taps[..1]).is_err());
    Ok(())
}

#[test]
fn m13_every_accelerated_kernel_reports_the_simd_backend() {
    let accelerated = [
        "convert",
        "convert_out",
        "rotate",
        "fir",
        "fir_decimate",
        "fir_interpolate",
        "resample",
        "mag_sq",
        "quad_demod",
        "fm_mod",
        "fft",
    ];
    // `simd` unless the feature is off, the build forces the fallback (`--cfg
    // sdrk_fallback_only`) or the target has no SIMD instructions.
    let expected = ACCEL;
    #[cfg(sdrk_fallback_only)]
    assert_eq!(expected, "fallback");
    #[cfg(all(
        not(sdrk_fallback_only),
        feature = "simd",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    assert_eq!(expected, "simd");
    for kernel in accelerated {
        for ty in [SampleType::C32, SampleType::CQ15] {
            assert_eq!(backend_for(kernel, ty), expected, "{kernel} {ty:?}");
        }
    }
    // Everything else stays with the fallback or the chained fused kernels.
    for (kernel, ty) in [
        ("scale", SampleType::C32),
        ("window", SampleType::C32),
        ("deemphasis", SampleType::F32),
        ("dot", SampleType::C32),
    ] {
        assert_eq!(backend_for(kernel, ty), "fallback", "{kernel} {ty:?}");
    }
    assert_eq!(backend_for("ddc", SampleType::C32), "chained");
    // The real FFTs run on the backend of the complex FFT.
    assert_eq!(backend_for("rfft", SampleType::F32), expected);
    assert_eq!(backend_for("irfft", SampleType::F32), expected);
}

#[test]
fn m20_kernels_agree_with_the_fallback() -> Result<()> {
    use sdrk::Rate;
    use sdrk::kernels::agc::{Agc, AgcState};
    use sdrk::kernels::demod::{DcBlockReal, DcBlockRealState};
    use sdrk::kernels::spectrum::Mag;

    let mut s = 20u64;
    let c32: Vec<C32> = (0..300)
        .map(|_| {
            C32::new(
                lcg(&mut s) as f32 / 4e9 - 0.5,
                lcg(&mut s) as f32 / 4e9 - 0.5,
            )
        })
        .collect();
    let cq: Vec<CQ15> = (0..300)
        .map(|_| CQ15 {
            re: Q15(lcg(&mut s) as i16),
            im: Q15(lcg(&mut s) as i16),
        })
        .collect();

    let (mut a, mut b) = (vec![0.0f32; 300], vec![0.0f32; 300]);
    Sdrk.mag(&c32, &mut a)?;
    Fallback.mag(&c32, &mut b)?;
    assert_eq!(a, b);
    let (mut a, mut b) = (vec![Q15(0); 300], vec![Q15(0); 300]);
    Sdrk.mag(&cq, &mut a)?;
    Fallback.mag(&cq, &mut b)?;
    assert_eq!(a, b);

    let r = Rate::hz(48_000)?;
    let new32 = || AgcState::<C32>::new(0.5, 0.005, 0.05, -20.0, 60.0, 0.0, r);
    let newq = || AgcState::<CQ15>::new(0.5, 0.005, 0.05, -20.0, 60.0, 0.0, r);
    let (mut sa, mut sb) = (new32()?, new32()?);
    let (mut a, mut b) = (vec![C32::default(); 300], vec![C32::default(); 300]);
    Sdrk.agc(&mut sa, &c32, &mut a)?;
    Fallback.agc(&mut sb, &c32, &mut b)?;
    assert_eq!((a, sa), (b, sb));
    let (mut sa, mut sb) = (newq()?, newq()?);
    let (mut a, mut b) = (vec![CQ15::default(); 300], vec![CQ15::default(); 300]);
    Sdrk.agc(&mut sa, &cq, &mut a)?;
    Fallback.agc(&mut sb, &cq, &mut b)?;
    assert_eq!((a, sa), (b, sb));

    let re: Vec<f32> = c32.iter().map(|v| v.re).collect();
    let rq: Vec<Q15> = cq.iter().map(|v| v.re).collect();
    let (mut sa, mut sb) = (
        DcBlockRealState::<f32>::new(0.99)?,
        DcBlockRealState::<f32>::new(0.99)?,
    );
    let (mut a, mut b) = (vec![0.0f32; 300], vec![0.0f32; 300]);
    Sdrk.dc_block_real(&mut sa, &re, &mut a)?;
    Fallback.dc_block_real(&mut sb, &re, &mut b)?;
    assert_eq!((a, sa), (b, sb));
    let (mut sa, mut sb) = (
        DcBlockRealState::<Q15>::new(Q15(32000))?,
        DcBlockRealState::<Q15>::new(Q15(32000))?,
    );
    let (mut a, mut b) = (vec![Q15(0); 300], vec![Q15(0); 300]);
    Sdrk.dc_block_real(&mut sa, &rq, &mut a)?;
    Fallback.dc_block_real(&mut sb, &rq, &mut b)?;
    assert_eq!((a, sa), (b, sb));
    Ok(())
}
