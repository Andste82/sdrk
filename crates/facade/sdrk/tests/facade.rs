//! Integration tests of the facade `Sdrk`.

#![allow(
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    clippy::items_after_statements
)]
use sdrk::kernels::basic::Scale;
use sdrk::{C32, CQ15, Error, Fallback, Q15, Result, SampleType, Sdrk, backend_for};

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
        ("convert_real", SampleType::F32),
        ("convert_real", SampleType::Q15),
        ("rotate", SampleType::C32),
        ("rotate", SampleType::CQ15),
        ("fir", SampleType::C32),
        ("fir", SampleType::CQ15),
    ] {
        assert_eq!(backend_for(kernel, ty), "fallback", "{kernel} {ty:?}");
    }
    assert_eq!(backend_for("convert", SampleType::F32), "none");
    assert_eq!(backend_for("convert_real", SampleType::C32), "none");
    assert_eq!(backend_for("rotate", SampleType::Q15), "none");
    assert_eq!(backend_for("fir", SampleType::F32), "none");
}

#[test]
fn m05_kernels_delegate_to_the_fallback() -> Result<()> {
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
    assert_eq!((a, na), (b, nb));

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
    assert_eq!((a, sa.history()), (b, sb.history()));

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
