//! Tests of the M22 items of `sdrk-core`: the construction rules of the biquad states,
//! `q14_from_f64`, timing, hand-computed responses, the flush schedule of the `f32` states
//! (decision D17) and the thread-safety of the public types.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::items_after_statements,
    clippy::indexing_slicing
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

// The flush schedule of the `f32` states (decision D17): section `k` flushes both states after
// sample `n` when `(n + k) % FLUSH_PERIOD == FLUSH_PERIOD - 1`, `n` counted since `new` or
// `reset`; the phase is part of the state. The inputs below drive the states into the flush
// range (0, 1e-30), where a flush changes a value.

use sdrk_core::kernels::biquad::FLUSH_PERIOD;

/// Spacing of the impulses of [`impulses`]: the response of 32 sections decays below 1e-30
/// well within it.
const SPACING: usize = 2000;

/// The two sections of the fourth-order Butterworth low-pass at 0.1 of the sample rate,
/// alternating over `sections`.
fn cascade(sections: usize) -> Vec<[f32; 5]> {
    let k = (core::f64::consts::PI * 0.1).tan();
    let section = |d: f64| {
        let n = 1.0 / (1.0 + d * k + k * k);
        [
            k * k * n,
            2.0 * k * k * n,
            k * k * n,
            2.0 * (k * k - 1.0) * n,
            (1.0 - d * k + k * k) * n,
        ]
        .map(|v| v as f32)
    };
    let pair = [
        section(2.0 * (core::f64::consts::PI / 8.0).cos()),
        section(2.0 * (3.0 * core::f64::consts::PI / 8.0).cos()),
    ];
    (0..sections).map(|k| pair[k % 2]).collect()
}

/// A small deterministic generator for the call lengths (xorshift64*).
struct Lengths(u64);

impl Lengths {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 33) as usize) % n.max(1)
    }

    /// Splits `len` samples into calls of the lengths 0, 1, 15, 16 and 17 in turn with random
    /// lengths of 0 to 40.
    fn split(&mut self, len: usize) -> Vec<usize> {
        const FIXED: [usize; 5] = [0, 1, 15, 16, 17];
        let mut out = Vec::new();
        let mut left = len;
        let mut i = 0;
        while left > 0 {
            let c = if i % 2 == 0 {
                FIXED[(i / 2) % FIXED.len()]
            } else {
                self.below(41)
            };
            let c = c.min(left);
            out.push(c);
            left -= c;
            i += 1;
        }
        out
    }
}

/// Impulses of `scale` every [`SPACING`] samples followed by silence: every impulse response
/// decays into the flush range.
fn impulses(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| if i % SPACING == 3 { scale } else { 0.0 })
        .collect()
}

/// The states `(s1, s2)` of section `k` of an `f32` history.
fn states_f32(hist: &[f32], k: usize) -> [f32; 2] {
    [hist[4 * k], hist[4 * k + 1]]
}

/// The states of section `k` of a `C32` history: `(s1, s2)` of re, then of im.
fn states_c32(hist: &[C32], k: usize) -> [f32; 4] {
    [
        hist[4 * k].re,
        hist[4 * k + 1].re,
        hist[4 * k].im,
        hist[4 * k + 1].im,
    ]
}

#[test]
fn f32_flush_active_input_in_chunks_equals_one_call() {
    let mut rng = Lengths(0x9e37_79b9_7f4a_7c15);
    for sections in [1, 2, 3, 31, MAX_SECTIONS] {
        let coefs = cascade(sections);
        let x = impulses(3 * SPACING, 1.0);
        let new = || BiquadRealState::<f32, _, _>::new(coefs.clone(), vec![0.0f32; 4 * sections]);
        let (mut whole, mut parts) = (new().unwrap(), new().unwrap());
        let mut want = vec![0.0f32; x.len()];
        Fallback.biquad_real(&mut whole, &x, &mut want).unwrap();
        let mut got = vec![0.0f32; x.len()];
        let mut done = 0;
        for c in rng.split(x.len()) {
            let r = done..done + c;
            Fallback
                .biquad_real(&mut parts, &x[r.clone()], &mut got[r])
                .unwrap();
            done += c;
        }
        let bits = |v: &[f32]| v.iter().map(|s| s.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&want), bits(&got), "{sections} sections");
        assert_eq!(bits(whole.history()), bits(parts.history()));
        assert_eq!(whole.phase(), parts.phase());
        // The last impulse has decayed and the flush set every state to zero (without it the
        // states would stay in the flush range).
        assert!(
            whole.history().iter().all(|&s| s == 0.0),
            "{sections} sections"
        );
    }
}

#[test]
fn c32_flush_active_input_in_chunks_equals_one_call() {
    let mut rng = Lengths(0x2545_f491_4f6c_dd1d);
    for sections in [1, 2, 3, 31, MAX_SECTIONS] {
        let coefs = cascade(sections);
        let x: Vec<C32> = impulses(3 * SPACING, 1.0)
            .iter()
            .map(|&v| C32::new(v, -1e-20 * v))
            .collect();
        let new =
            || BiquadState::<C32, _, _>::new(coefs.clone(), vec![C32::default(); 4 * sections]);
        let (mut whole, mut parts) = (new().unwrap(), new().unwrap());
        let mut want = vec![C32::default(); x.len()];
        Fallback.biquad(&mut whole, &x, &mut want).unwrap();
        let mut got = vec![C32::default(); x.len()];
        let mut done = 0;
        for c in rng.split(x.len()) {
            let r = done..done + c;
            Fallback
                .biquad(&mut parts, &x[r.clone()], &mut got[r])
                .unwrap();
            done += c;
        }
        let bits = |v: &[C32]| {
            v.iter()
                .map(|s| (s.re.to_bits(), s.im.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&want), bits(&got), "{sections} sections");
        assert_eq!(bits(whole.history()), bits(parts.history()));
        assert_eq!(whole.phase(), parts.phase());
        assert!(
            whole.history().iter().all(|&s| s == C32::default()),
            "{sections} sections"
        );
    }
}

/// Runs `len` steps (`step(n)` processes sample `n` and returns the states of every section)
/// and checks after every sample `n` and for every section `k` that normal states turn to exact
/// zeros only after the samples of the schedule of `k`, and that no state is left in the flush
/// range after such a sample. Returns, per section, how often the flush turned normal states
/// to zero.
fn check_schedule<const W: usize>(
    sections: usize,
    len: usize,
    mut step: impl FnMut(usize) -> Vec<[f32; W]>,
) -> Vec<usize> {
    let mut zeroed = vec![0usize; sections];
    let mut before = vec![[0.0f32; W]; sections];
    for n in 0..len {
        let after = step(n);
        for k in 0..sections {
            let was_zero = before[k].iter().all(|&s| s == 0.0);
            let is_zero = after[k].iter().all(|&s| s == 0.0);
            let scheduled = (n + k) % FLUSH_PERIOD == FLUSH_PERIOD - 1;
            // Outside the schedule, only subnormal states can turn to zero (by the underflow of
            // the arithmetic itself); a normal state turns to zero only by the flush.
            let normal = before[k].iter().any(|&s| s.abs() >= f32::MIN_POSITIVE);
            if is_zero && !was_zero {
                assert!(
                    scheduled || !normal,
                    "section {k} became zero after sample {n}: {:?}",
                    before[k]
                );
                if normal {
                    zeroed[k] += 1;
                }
            }
            if scheduled {
                assert!(
                    after[k].iter().all(|&s| s == 0.0 || s.abs() >= 1e-30),
                    "section {k} kept a state in the flush range after sample {n}: {:?}",
                    after[k]
                );
            }
        }
        before = after;
    }
    zeroed
}

#[test]
fn f32_states_become_zero_on_the_schedule_of_each_section() {
    for sections in [1, 2, 3, MAX_SECTIONS] {
        let mut state =
            BiquadRealState::<f32, _, _>::new(cascade(sections), vec![0.0f32; 4 * sections])
                .unwrap();
        let x = impulses(2 * SPACING, 1.0);
        let zeroed = check_schedule(sections, x.len(), |n| {
            let mut y = [0.0f32];
            Fallback.biquad_real(&mut state, &x[n..=n], &mut y).unwrap();
            assert_eq!(state.phase(), (n + 1) % FLUSH_PERIOD);
            (0..sections)
                .map(|k| states_f32(state.history(), k))
                .collect()
        });
        // Every section decayed into the flush range and was set to zero by its flush after
        // each of the two impulses.
        assert!(
            zeroed.iter().all(|&z| z >= 2),
            "{sections} sections: {zeroed:?}"
        );
    }
}

#[test]
fn c32_states_become_zero_on_the_schedule_of_each_section() {
    for sections in [1, 2, 3, MAX_SECTIONS] {
        let mut state =
            BiquadState::<C32, _, _>::new(cascade(sections), vec![C32::default(); 4 * sections])
                .unwrap();
        let x: Vec<C32> = impulses(2 * SPACING, 1.0)
            .iter()
            .map(|&v| C32::new(v, -0.25 * v))
            .collect();
        let zeroed = check_schedule(sections, x.len(), |n| {
            let mut y = [C32::default()];
            Fallback.biquad(&mut state, &x[n..=n], &mut y).unwrap();
            assert_eq!(state.phase(), (n + 1) % FLUSH_PERIOD);
            (0..sections)
                .map(|k| states_c32(state.history(), k))
                .collect()
        });
        assert!(
            zeroed.iter().all(|&z| z >= 2),
            "{sections} sections: {zeroed:?}"
        );
    }
}

#[test]
fn set_coefs_keeps_the_phase_and_reset_clears_it() {
    let mut f = BiquadRealState::<f32, _, _>::new(cascade(3), vec![0.0f32; 12]).unwrap();
    let mut c = BiquadState::<C32, _, _>::new(cascade(3), vec![C32::default(); 12]).unwrap();
    assert_eq!((f.phase(), c.phase()), (0, 0));
    let x = [0.5f32; 37];
    let xc = [C32::new(0.5, -0.5); 37];
    Fallback.biquad_real(&mut f, &x, &mut [0.0; 37]).unwrap();
    Fallback
        .biquad(&mut c, &xc, &mut [C32::default(); 37])
        .unwrap();
    assert_eq!(
        (f.phase(), c.phase()),
        (37 % FLUSH_PERIOD, 37 % FLUSH_PERIOD)
    );
    // A call of no samples changes nothing.
    Fallback.biquad_real(&mut f, &[], &mut []).unwrap();
    Fallback.biquad(&mut c, &[], &mut []).unwrap();
    assert_eq!((f.phase(), c.phase()), (5, 5));
    // Retuning keeps the phase and the states.
    let (hf, hc) = (f.history().to_vec(), c.history().to_vec());
    let other: Vec<[f32; 5]> = cascade(3).into_iter().rev().collect();
    f.set_coefs(other.clone()).unwrap();
    c.set_coefs(other).unwrap();
    assert_eq!((f.phase(), c.phase()), (5, 5));
    assert_eq!(f.history(), &hf[..]);
    assert_eq!(c.history(), &hc[..]);
    // The schedule continues: eleven more samples complete the period.
    Fallback
        .biquad_real(&mut f, &x[..11], &mut [0.0; 11])
        .unwrap();
    Fallback
        .biquad(&mut c, &xc[..11], &mut [C32::default(); 11])
        .unwrap();
    assert_eq!((f.phase(), c.phase()), (0, 0));
    Fallback
        .biquad_real(&mut f, &x[..3], &mut [0.0; 3])
        .unwrap();
    Fallback
        .biquad(&mut c, &xc[..3], &mut [C32::default(); 3])
        .unwrap();
    assert_eq!((f.phase(), c.phase()), (3, 3));
    // Reset clears the phase and the states.
    f.reset();
    c.reset();
    assert_eq!((f.phase(), c.phase()), (0, 0));
    assert!(f.history().iter().all(|&s| s == 0.0));
    assert!(c.history().iter().all(|&s| s == C32::default()));
}
