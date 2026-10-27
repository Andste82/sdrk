//! Tests of the M22 items of `sdrk-core`: the construction rules of the biquad states,
//! `q14_from_f64`, timing, hand-computed responses and the thread-safety of the public types.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::items_after_statements
)]

use sdrk_core::kernels::biquad::{
    Biquad, BiquadReal, BiquadRealState, BiquadSample, BiquadState, MAX_SECTIONS,
};
use sdrk_core::num::q14_from_f64;
use sdrk_core::{C32, CQ15, Delay, Error, Fallback, Q15, Ratio, Timing};
use sdrk_device::{ErrorKind, SdrkError};

const IDENTITY_F: [f32; 5] = [1.0, 0.0, 0.0, 0.0, 0.0];
const IDENTITY_Q: [i16; 5] = [16384, 0, 0, 0, 0];

#[test]
fn q14_rounds_half_up_and_rejects_the_overflow() {
    assert_eq!(q14_from_f64(0.0), Some(0));
    assert_eq!(q14_from_f64(1.0), Some(16384));
    assert_eq!(q14_from_f64(-2.0), Some(-32768));
    assert_eq!(q14_from_f64(2.0), None);
    assert_eq!(q14_from_f64(-2.0 - 1.0 / 16384.0), None);
    // The largest accepted value is 32767 / 16384; a half LSB above it rounds to 32768.
    assert_eq!(q14_from_f64(32767.0 / 16384.0), Some(32767));
    assert_eq!(q14_from_f64(32767.4 / 16384.0), Some(32767));
    assert_eq!(q14_from_f64(32767.5 / 16384.0), None);
    // Ties go up, also for negative values.
    assert_eq!(q14_from_f64(0.5 / 16384.0), Some(1));
    assert_eq!(q14_from_f64(-0.5 / 16384.0), Some(0));
    assert_eq!(q14_from_f64(-1.5 / 16384.0), Some(-1));
    assert_eq!(q14_from_f64(f64::NAN), None);
    assert_eq!(q14_from_f64(f64::INFINITY), None);
    assert_eq!(q14_from_f64(f64::NEG_INFINITY), None);
}

#[test]
fn the_section_count_must_be_between_1_and_32() {
    let make =
        |n: usize| BiquadRealState::<f32, _, _>::new(vec![IDENTITY_F; n], vec![0.0f32; 4 * n]);
    assert!(matches!(make(0), Err(Error::InvalidParameter(_))));
    for n in [1, 2, 31, MAX_SECTIONS] {
        assert_eq!(make(n).unwrap().sections(), n);
    }
    assert!(matches!(
        make(MAX_SECTIONS + 1),
        Err(Error::InvalidParameter(_))
    ));
    let make = |n: usize| {
        BiquadState::<CQ15, _, _>::new(vec![IDENTITY_Q; n], vec![CQ15::default(); 4 * n])
    };
    assert!(matches!(make(0), Err(Error::InvalidParameter(_))));
    assert_eq!(make(MAX_SECTIONS).unwrap().sections(), MAX_SECTIONS);
    assert!(make(MAX_SECTIONS + 1).is_err());
}

#[test]
fn the_history_needs_four_entries_per_section() {
    for got in [0, 1, 3, 5, 8] {
        let r = BiquadState::<C32, _, _>::new([IDENTITY_F], vec![C32::default(); got]);
        assert_eq!(r.unwrap_err(), Error::LengthMismatch { expected: 4, got });
    }
    let r = BiquadRealState::<Q15, _, _>::new([IDENTITY_Q; 2], [Q15(0); 4]);
    let e = r.unwrap_err();
    assert_eq!(
        e,
        Error::LengthMismatch {
            expected: 8,
            got: 4
        }
    );
    assert_eq!(e.kind(), ErrorKind::LengthMismatch);
    // A history of the right length is zeroed, whatever it held.
    let s = BiquadRealState::<Q15, _, _>::new([IDENTITY_Q], [Q15(7); 4]).unwrap();
    assert_eq!(s.history(), [Q15(0); 4]);
}

#[test]
fn storage_can_be_an_array_a_slice_or_a_vec() {
    let mut hist = [0.0f32; 4];
    let coefs = [IDENTITY_F];
    let mut s =
        BiquadRealState::<f32, &[[f32; 5]], &mut [f32]>::new(&coefs, &mut hist[..]).unwrap();
    let mut out = [0.0f32; 2];
    Fallback.biquad_real(&mut s, &[1.0, 2.0], &mut out).unwrap();
    assert_eq!(out, [1.0, 2.0]);
}

#[test]
fn set_coefs_keeps_the_history_and_checks_the_section_count() {
    let lp = [0.25f32, 0.0, 0.0, -0.75, 0.0]; // y = 0.25 x + 0.75 y1
    let mut s = BiquadRealState::<f32, _, _>::new(vec![lp], [0.0f32; 4]).unwrap();
    let mut out = [0.0f32; 2];
    Fallback.biquad_real(&mut s, &[1.0, 1.0], &mut out).unwrap();
    let kept = s.history().to_vec();
    assert!(kept.iter().any(|&v| v != 0.0));
    // A wrong number of sections is rejected and changes nothing.
    let e = s.set_coefs(vec![lp, lp]).unwrap_err();
    assert_eq!(
        e,
        Error::LengthMismatch {
            expected: 1,
            got: 2
        }
    );
    assert_eq!(s.coefs(), [lp]);
    assert_eq!(s.history(), kept);
    // Retuning keeps the history.
    s.set_coefs(vec![IDENTITY_F]).unwrap();
    assert_eq!(s.coefs(), [IDENTITY_F]);
    assert_eq!(s.history(), kept);
    s.reset();
    assert_eq!(s.history(), [0.0; 4]);
    assert_eq!(s.coefs(), [IDENTITY_F]);
}

#[test]
fn the_state_has_no_delay_and_ratio_one() {
    let s = BiquadState::<C32, _, _>::new([IDENTITY_F], [C32::default(); 4]).unwrap();
    assert_eq!(s.ratio(), Ratio::ONE);
    assert_eq!(s.group_delay(), Delay::ZERO);
    let s = BiquadRealState::<Q15, _, _>::new([IDENTITY_Q], [Q15(0); 4]).unwrap();
    assert_eq!(s.ratio(), Ratio::ONE);
    assert_eq!(s.group_delay(), Delay::ZERO);
}

#[test]
fn lengths_must_match() {
    let mut s = BiquadRealState::<f32, _, _>::new([IDENTITY_F], [0.0f32; 4]).unwrap();
    let mut out = [0.0f32; 3];
    let e = Fallback
        .biquad_real(&mut s, &[0.0; 2], &mut out)
        .unwrap_err();
    assert_eq!(
        e,
        Error::LengthMismatch {
            expected: 2,
            got: 3
        }
    );
    let mut s = BiquadState::<CQ15, _, _>::new([IDENTITY_Q], [CQ15::default(); 4]).unwrap();
    let mut out = [CQ15::default(); 1];
    assert!(
        Fallback
            .biquad(&mut s, &[CQ15::default(); 2], &mut out)
            .is_err()
    );
}

#[test]
fn f32_impulse_response_of_a_first_order_like_section() {
    // y[n] = x[n] + 0.5 y[n-1]: h[n] = 0.5^n (b = [1, 0, 0], a = [-0.5, 0]).
    let mut s =
        BiquadRealState::<f32, _, _>::new([[1.0, 0.0, 0.0, -0.5, 0.0]], [0.0f32; 4]).unwrap();
    let mut x = [0.0f32; 8];
    x[0] = 1.0;
    let mut y = [0.0f32; 8];
    Fallback.biquad_real(&mut s, &x, &mut y).unwrap();
    for (n, v) in y.iter().enumerate() {
        assert_eq!(*v, 0.5f32.powi(n as i32));
    }
}

#[test]
fn f32_second_order_matches_the_difference_equation() {
    let c = [0.2f32, 0.4, 0.2, -0.5, 0.25];
    let x: Vec<f32> = (0..16).map(|n| ((n * 7 % 5) as f32 - 2.0) * 0.25).collect();
    let mut want = vec![0.0f64; x.len()];
    for n in 0..x.len() {
        let xv = |k: usize| if n >= k { f64::from(x[n - k]) } else { 0.0 };
        let yv = |k: usize| if n >= k { want[n - k] } else { 0.0 };
        want[n] = f64::from(c[0]) * xv(0) + f64::from(c[1]) * xv(1) + f64::from(c[2]) * xv(2)
            - f64::from(c[3]) * yv(1)
            - f64::from(c[4]) * yv(2);
    }
    let mut s = BiquadRealState::<f32, _, _>::new([c], [0.0f32; 4]).unwrap();
    let mut y = vec![0.0f32; x.len()];
    Fallback.biquad_real(&mut s, &x, &mut y).unwrap();
    for (got, want) in y.iter().zip(&want) {
        assert!((f64::from(*got) - want).abs() < 1e-6, "{got} {want}");
    }
}

#[test]
fn sections_run_in_series() {
    // Two sections: gain 0.5 then a one-sample delay; the order does not change the result of
    // linear sections, so check against the product of both.
    let gain = [0.5f32, 0.0, 0.0, 0.0, 0.0];
    let delay = [0.0f32, 1.0, 0.0, 0.0, 0.0];
    let mut s = BiquadRealState::<f32, _, _>::new([gain, delay], [0.0f32; 8]).unwrap();
    let mut y = [0.0f32; 4];
    Fallback
        .biquad_real(&mut s, &[1.0, 2.0, 3.0, 4.0], &mut y)
        .unwrap();
    assert_eq!(y, [0.0, 0.5, 1.0, 1.5]);
}

#[test]
fn complex_components_are_filtered_independently() {
    let lp = [0.25f32, 0.0, 0.0, -0.75, 0.0];
    let mut c = BiquadState::<C32, _, _>::new([lp], [C32::default(); 4]).unwrap();
    let mut re = BiquadRealState::<f32, _, _>::new([lp], [0.0f32; 4]).unwrap();
    let mut im = BiquadRealState::<f32, _, _>::new([lp], [0.0f32; 4]).unwrap();
    let x: Vec<C32> = (0..50)
        .map(|n| C32::new((n as f32 * 0.37).sin(), (n as f32 * 0.11).cos()))
        .collect();
    let mut y = vec![C32::default(); x.len()];
    Fallback.biquad(&mut c, &x, &mut y).unwrap();
    let (xr, xi): (Vec<f32>, Vec<f32>) = x.iter().map(|v| (v.re, v.im)).unzip();
    let (mut yr, mut yi) = (vec![0.0f32; x.len()], vec![0.0f32; x.len()]);
    Fallback.biquad_real(&mut re, &xr, &mut yr).unwrap();
    Fallback.biquad_real(&mut im, &xi, &mut yi).unwrap();
    for ((v, r), i) in y.iter().zip(&yr).zip(&yi) {
        assert_eq!((v.re, v.im), (*r, *i));
    }
}

#[test]
fn q15_direct_form_one_rounds_half_up_and_saturates() {
    // y = sat16(round_shift(b0 x, 14)) with b0 = 0.5 (8192): 3 * 0.5 = 1.5 rounds to 2,
    // -3 * 0.5 = -1.5 rounds to -1.
    let mut s = BiquadRealState::<Q15, _, _>::new([[8192i16, 0, 0, 0, 0]], [Q15(0); 4]).unwrap();
    let mut y = [Q15(0); 2];
    Fallback
        .biquad_real(&mut s, &[Q15(3), Q15(-3)], &mut y)
        .unwrap();
    assert_eq!(y, [Q15(2), Q15(-1)]);
    // A gain of 2 - 2^-14 saturates.
    let mut s = BiquadRealState::<Q15, _, _>::new([[32767i16, 0, 0, 0, 0]], [Q15(0); 4]).unwrap();
    let mut y = [Q15(0); 2];
    Fallback
        .biquad_real(&mut s, &[Q15(30000), Q15(-30000)], &mut y)
        .unwrap();
    assert_eq!(y, [Q15(32767), Q15(-32768)]);
    // The history holds x1, x2, y1, y2.
    assert_eq!(
        s.history(),
        [Q15(-30000), Q15(30000), Q15(-32768), Q15(32767)]
    );
}

#[cfg(feature = "reference")]
#[test]
fn reference_and_fallback_agree_on_q15_with_a_shared_state_layout() {
    use sdrk_core::Reference;
    let c = [
        [3000i16, -5000, 3000, -20000, 8000],
        [9000, 4000, 1000, -3000, 500],
    ];
    let x: Vec<Q15> = (0..400)
        .map(|n| Q15(((n * 7919) % 65536 - 32768) as i16))
        .collect();
    let mut a = BiquadRealState::<Q15, _, _>::new(c, [Q15(0); 8]).unwrap();
    let mut b = BiquadRealState::<Q15, _, _>::new(c, [Q15(0); 8]).unwrap();
    let (mut ya, mut yb) = (vec![Q15(0); 400], vec![Q15(0); 400]);
    Fallback.biquad_real(&mut a, &x, &mut ya).unwrap();
    Reference.biquad_real(&mut b, &x, &mut yb).unwrap();
    assert_eq!(ya, yb);
    assert_eq!(a.history(), b.history());
}

#[cfg(feature = "reference")]
#[test]
fn reference_f32_stays_close_to_the_fallback() {
    use sdrk_core::Reference;
    let c = [
        [0.0675f32, 0.135, 0.0675, -1.143, 0.4128],
        [0.1, 0.2, 0.1, -0.9, 0.3],
    ];
    let x: Vec<f32> = (0..2000).map(|n| ((n as f32) * 0.31).sin() * 0.5).collect();
    let mut a = BiquadRealState::<f32, _, _>::new(c, [0.0f32; 8]).unwrap();
    let mut b = BiquadRealState::<f32, _, _>::new(c, [0.0f32; 8]).unwrap();
    let (mut ya, mut yb) = (vec![0.0f32; x.len()], vec![0.0f32; x.len()]);
    Fallback.biquad_real(&mut a, &x, &mut ya).unwrap();
    Reference.biquad_real(&mut b, &x, &mut yb).unwrap();
    let err: f64 = ya
        .iter()
        .zip(&yb)
        .map(|(p, q)| f64::from(p - q).powi(2))
        .sum();
    let pow: f64 = yb.iter().map(|q| f64::from(*q).powi(2)).sum();
    assert!(10.0 * (pow / err).log10() > 100.0);
}

#[test]
fn the_public_types_are_send_and_sync() {
    fn assert_send<T: Send + Sync>() {}
    assert_send::<BiquadState<C32, Vec<[f32; 5]>, Vec<C32>>>();
    assert_send::<BiquadState<CQ15, [[i16; 5]; 2], [CQ15; 8]>>();
    assert_send::<BiquadRealState<f32, &'static [[f32; 5]], Vec<f32>>>();
    assert_send::<BiquadRealState<Q15, Vec<[i16; 5]>, Vec<Q15>>>();
    fn coef<T: BiquadSample>() -> usize {
        core::mem::size_of::<T::Coef>()
    }
    assert_eq!((coef::<C32>(), coef::<f32>()), (4, 4));
    assert_eq!((coef::<CQ15>(), coef::<Q15>()), (2, 2));
}
