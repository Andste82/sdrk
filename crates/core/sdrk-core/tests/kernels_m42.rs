//! Tests of the M42 kernels of `sdrk-core`: `rds_shape` and `stereo_encode_rds`. The pulse table
//! (odd, peak 1, the end point), the construction rules, timing, length checks, the timing of the
//! bit requests, the single pulse, the RDS term of the encoder and the thread-safety of the
//! public types. The comparison with `Reference` and the golden cases are in the testkit
//! (`golden_m42.rs`, suite cases `rds_shape_f32`, `stereo_encode_rds_c32`, `extremes_m42`).

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing
)]

use sdrk_core::kernels::rds::{
    RDS_TABLE_LEN, RDS_TABLE_PER_BIT, RdsShape, RdsShapeState, rds_pulse_table, rds_pulse_table_f64,
};
use sdrk_core::kernels::stereo::{
    StereoEncode, StereoEncodeRds, StereoEncodeRdsState, StereoEncodeState,
};
use sdrk_core::{C32, Delay, Error, Fallback, Rate, Ratio, Timing};

fn rate(hz: u64) -> Rate {
    Rate::hz(hz).unwrap()
}

fn table() -> Vec<f32> {
    let mut t = vec![0.0f32; RDS_TABLE_LEN];
    rds_pulse_table(&mut t).unwrap();
    t
}

#[test]
fn the_pulse_table_is_odd_with_peak_one() {
    let mut t = vec![0.0f64; RDS_TABLE_LEN];
    rds_pulse_table_f64(&mut t).unwrap();
    assert_eq!(RDS_TABLE_LEN, 4 * RDS_TABLE_PER_BIT + 1);
    let mid = 2 * RDS_TABLE_PER_BIT;
    for i in 0..=mid {
        assert!((t[mid + i] + t[mid - i]).abs() < 1e-14, "i={i}");
    }
    let (imax, vmax) = t
        .iter()
        .enumerate()
        .fold((0, 0.0f64), |m, (i, &v)| if v > m.1 { (i, v) } else { m });
    assert_eq!(vmax, 1.0);
    // The peak lies in the first chip (t < 0), near -Tb/4.
    assert!(
        imax < mid && imax.abs_diff(mid - RDS_TABLE_PER_BIT / 4) < 16,
        "{imax}"
    );
    // The ends are small but not zero (p(+-2 Tb)).
    assert!(t[0].abs() < 0.02 && t[0] != 0.0);
    assert_eq!(t[RDS_TABLE_LEN - 1], -t[0]);
    assert!(matches!(
        rds_pulse_table_f64(&mut [0.0; 3]),
        Err(Error::LengthMismatch { .. })
    ));
}

#[test]
fn shaper_construction_and_timing() {
    let t = table();
    assert!(RdsShapeState::new(&t[..], rate(4_750)).is_ok());
    assert!(matches!(
        RdsShapeState::new(&t[..], rate(4_749)),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        RdsShapeState::new(&t[1..], rate(48_000)),
        Err(Error::InvalidParameter(_))
    ));
    let st = RdsShapeState::new(&t[..], rate(192_000)).unwrap();
    assert_eq!(st.ratio(), Ratio::ONE);
    assert_eq!(st.group_delay(), Delay::ZERO);
    assert_eq!(st.rate_hz(), 192_000.0);
    assert_eq!(st.table().len(), RDS_TABLE_LEN);
    assert_eq!((st.position(), st.next_bit()), (0, 0));
}

#[test]
fn bits_are_requested_when_they_enter_the_window() {
    let t = table();
    // 4750 Hz: exactly 4 samples per bit, so the window moves every 4 samples.
    let mut st = RdsShapeState::new(&t[..], rate(4_750)).unwrap();
    let mut calls = 0u32;
    let mut out = [0.0f32; 1];
    Fallback
        .rds_shape(
            &mut st,
            &mut || {
                calls += 1;
                1
            },
            &mut out,
        )
        .unwrap();
    // Bits 0, 1 and 2 before the first sample.
    assert_eq!(calls, 3);
    for n in 1..=12u32 {
        Fallback
            .rds_shape(
                &mut st,
                &mut || {
                    calls += 1;
                    1
                },
                &mut out,
            )
            .unwrap();
        assert_eq!(calls, 3 + n / 4, "n={n}");
    }
    assert_eq!(st.position(), 13);
    assert_eq!(st.next_bit(), 6);
    // An empty output fetches nothing.
    Fallback
        .rds_shape(
            &mut st,
            &mut || {
                calls += 1;
                1
            },
            &mut [],
        )
        .unwrap();
    assert_eq!(calls, 6);
}

#[test]
fn a_single_bit_is_the_pulse() {
    let t = table();
    // 304 kHz: exactly 256 samples per bit, so the output is the table itself.
    let fs = 1187.5 * 256.0;
    let mut st = RdsShapeState::new(&t[..], Rate::new(608_000, 2).unwrap()).unwrap();
    assert_eq!(st.rate_hz(), fs);
    // Bit 2 is +1, all others 0: its pulse starts at sample 0 (x = 0 is 2 Tb before its centre).
    let mut k = 0;
    let mut bits = || {
        k += 1;
        i8::from(k == 3)
    };
    let mut out = vec![0.0f32; 1024];
    Fallback.rds_shape(&mut st, &mut bits, &mut out).unwrap();
    assert_eq!(out, t[..1024]);
}

#[test]
fn the_encoder_adds_the_rds_term_last() {
    let fs = rate(192_000);
    assert!(StereoEncodeRdsState::new(19_000.0, 0.87, 0.09, -0.01, fs).is_err());
    assert!(StereoEncodeRdsState::new(19_000.0, 0.87, 0.09, 0.21, fs).is_err());
    assert!(StereoEncodeRdsState::new(17_000.0, 0.87, 0.09, 0.04, fs).is_err());
    assert!(StereoEncodeRdsState::new(19_000.0, 0.87, 0.09, 0.04, rate(96_000)).is_err());
    let mut st = StereoEncodeRdsState::new(19_000.0, 0.87, 0.09, 0.04, fs).unwrap();
    assert_eq!(st.ratio(), Ratio::ONE);
    assert_eq!(st.group_delay(), Delay::ZERO);
    st.set_pilot(0.1).unwrap();
    assert!(st.set_pilot(0.3).is_err());
    assert_eq!(st.pilot(), 0.1);
    assert!(st.set_rds(0.3).is_err());
    assert_eq!(st.rds(), 0.04);
    let input = vec![C32::new(0.25, -0.5); 300];
    let rds = vec![1.0f32; 300];
    let mut out = vec![0.0f32; 300];
    assert!(matches!(
        Fallback.stereo_encode_rds(&mut st, &input, &rds[1..], &mut out),
        Err(Error::LengthMismatch { .. })
    ));
    assert!(matches!(
        Fallback.stereo_encode_rds(&mut st, &input, &rds, &mut out[1..]),
        Err(Error::LengthMismatch { .. })
    ));
    assert_eq!(st.phase(), 0);
    Fallback
        .stereo_encode_rds(&mut st, &input, &rds, &mut out)
        .unwrap();
    let mut plain = StereoEncodeState::new(19_000.0, 0.87, 0.1, fs).unwrap();
    let mut base = vec![0.0f32; 300];
    Fallback
        .stereo_encode(&mut plain, &input, &mut base)
        .unwrap();
    assert_eq!(st.phase(), plain.phase());
    // RDS b = 1: the difference is 0.04 sin(3 phi); sin(3 phi) at 57 kHz has the period
    // 192 / 57 samples, so its sign changes within every 4 samples.
    for (i, (y, b)) in out.iter().zip(&base).enumerate() {
        let phi = 2.0 * core::f64::consts::PI * (i as f64 * 19_000.0 / 192_000.0);
        let want = 0.04 * libm::sin(3.0 * phi);
        assert!((f64::from(y - b) - want).abs() < 2e-7, "i={i}");
    }
    st.reset();
    assert_eq!(st.phase(), 0);
    assert_eq!(st.encoder().main(), 0.87);
}

#[test]
fn the_public_types_are_send_and_sync() {
    fn assert_send<T: Send + Sync>() {}
    assert_send::<RdsShapeState<[f32; RDS_TABLE_LEN]>>();
    assert_send::<RdsShapeState<&[f32]>>();
    assert_send::<StereoEncodeRdsState>();
}
