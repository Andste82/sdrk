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
