//! Tests of the M36 kernels of `sdrk-core`: `preemph_limit_stereo` and `stereo_encode`. The
//! construction rules, timing, the stereo limiter against the mono kernel of M25 (bit for bit
//! with equal channels), the linked gain, non-finite input, the limit at 20 times full scale,
//! the encoder against the composition of the M30 kernels `oscillate`, `mul_real` and
//! `add_real` (bit for bit), the pilot switched off, block-size invariance and the
//! thread-safety of the public types. The comparison of `Fallback` with `Reference` runs only
//! with the feature `reference`.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::manual_midpoint
)]

use sdrk_core::kernels::math::{AddReal, MulReal};
use sdrk_core::kernels::mix::{Nco, Oscillate};
use sdrk_core::kernels::preemph_limit::{
    PreemphLimit, PreemphLimitState, PreemphLimitStereo, PreemphLimitStereoState,
};
use sdrk_core::kernels::stereo::{StereoEncode, StereoEncodeState};
use sdrk_core::{C32, Delay, Error, Fallback, Rate, Ratio, Timing};

const PASS: [f64; 3] = [1.0, 0.0, 0.0];

fn rate(hz: u64) -> Rate {
    Rate::hz(hz).unwrap()
}

/// The design of `sdrk_blocks::design::preemphasis_coefs`, written out (as in `kernels_m25`).
fn preemph(tau_s: f64, fs: f64) -> [f64; 3] {
    let tau_p = 1.0 / (2.0 * core::f64::consts::PI * (fs / 2.4));
    let k = 2.0 * fs;
    let den = 1.0 + k * tau_p;
    [
        (1.0 + k * tau_s) / den,
        (1.0 - k * tau_s) / den,
        (1.0 - k * tau_p) / den,
    ]
}

/// Deterministic generator (xorshift64*), uniform in [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (v as f64 / (1u64 << 52) as f64 - 1.0) as f32
    }
}

fn stereo(coefs: [f64; 3], limit: f64, hold_s: f64, release_s: f64) -> PreemphLimitStereoState {
    PreemphLimitStereoState::new(coefs, limit, hold_s, release_s, rate(48_000)).unwrap()
}

fn run_stereo<K: PreemphLimitStereo>(
    k: &K,
    s: &mut PreemphLimitStereoState,
    input: &[C32],
) -> Vec<C32> {
    let mut out = vec![C32::new(0.0, 0.0); input.len()];
    k.preemph_limit_stereo(s, input, &mut out).unwrap();
    out
}

fn run_encode<K: StereoEncode>(k: &K, s: &mut StereoEncodeState, input: &[C32]) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    k.stereo_encode(s, input, &mut out).unwrap();
    out
}

fn bits(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
}

fn bits_real(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// A tone on the left, noise on the right, bursts of `loud` on the left.
fn audio(n: usize, seed: u64, loud: f32) -> Vec<C32> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|i| {
            let tone = 0.5 * (0.13 * i as f32).sin();
            let burst = if (i / 300) % 3 == 1 {
                loud * (1.3 * i as f32).sin()
            } else {
                0.0
            };
            C32::new(tone + burst, 0.3 * rng.next())
        })
        .collect()
}

#[allow(clippy::needless_pass_by_value)]
fn invalid(r: sdrk_core::Result<impl Sized>) -> bool {
    matches!(r, Err(Error::InvalidParameter(_)))
}

#[test]
fn stereo_limiter_construction_rules() {
    let r = rate(48_000);
    let ok = |c: [f64; 3], l: f64, h: f64, rel: f64| PreemphLimitStereoState::new(c, l, h, rel, r);
    assert!(ok(PASS, 1.0, 0.0, 0.2).is_ok());
    assert!(ok(preemph(50e-6, 48_000.0), 0.9, 0.02, 0.2).is_ok());
    for l in [0.0, -0.5, 1.0001, f64::NAN, f64::INFINITY] {
        assert!(invalid(ok(PASS, l, 0.02, 0.2)), "limit {l}");
    }
    for h in [-1e-9, f64::NAN, f64::INFINITY] {
        assert!(invalid(ok(PASS, 1.0, h, 0.2)), "hold {h}");
    }
    // 2^31 samples is the longest hold, one more is rejected.
    assert!(ok(PASS, 1.0, 2_147_483_648.0 / 48_000.0, 0.2).is_ok());
    assert!(invalid(ok(PASS, 1.0, 2_147_483_649.0 / 48_000.0, 0.2)));
    for rel in [0.0, -0.1, f64::NAN, f64::INFINITY, 1e4] {
        assert!(invalid(ok(PASS, 1.0, 0.0, rel)), "release {rel}");
    }
    for c in [
        [f64::NAN, 0.0, 0.0],
        [1.0, f64::INFINITY, 0.0],
        [1.0, 0.0, 16_384.0],
    ] {
        assert!(invalid(ok(c, 1.0, 0.0, 0.2)), "{c:?}");
    }
    // The same rules as the mono `f32` kernel.
    for (c, l, h, rel) in [
        (PASS, 0.5, 0.01, 700.0),
        (PASS, 0.5, 0.01, 800.0),
        (PASS, 1.0, 44_739.0, 0.2),
        (PASS, 1.0, 44_740.0, 0.2),
    ] {
        assert_eq!(
            ok(c, l, h, rel).is_ok(),
            PreemphLimitState::<f32>::new(c, l, h, rel, r).is_ok(),
            "{l} {h} {rel}"
        );
    }
}

#[test]
fn stereo_limiter_timing_accessors_and_lengths() {
    let mut s = stereo(PASS, 0.5, 0.02, 0.2);
    assert_eq!(s.ratio(), Ratio::ONE);
    assert_eq!(s.group_delay(), Delay::ZERO);
    assert_eq!((s.limit(), s.hold_samples(), s.gain()), (0.5, 960, 1.0));
    s.set_limit(0.25).unwrap();
    assert!(invalid(s.set_limit(0.0)));
    assert_eq!(s.limit(), 0.25);
    let before = s;
    let mut out = [C32::new(0.0, 0.0); 3];
    let e = Fallback
        .preemph_limit_stereo(&mut s, &[C32::new(1.0, 1.0); 2], &mut out)
        .unwrap_err();
    assert!(matches!(
        e,
        Error::LengthMismatch {
            expected: 2,
            got: 3
        }
    ));
    assert_eq!(s, before);
    run_stereo(&Fallback, &mut s, &[C32::new(1.0, -0.5)]);
    assert_eq!(s.gain(), 0.25);
    s.reset();
    assert_eq!(s.gain(), 1.0);
    assert_eq!(s, before);
}

/// With `re == im` each channel of the output is the mono kernel of M25 bit for bit, over
/// pre-emphasis designs, limits, hold and release times, loud passages and non-finite input.
#[test]
fn equal_channels_give_the_mono_kernel() {
    let mut rng = Rng(36);
    let specials = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e30, -3000.0];
    for round in 0..60 {
        let fs = [44_100u64, 48_000, 192_000][round % 3];
        let c = match round % 4 {
            0 => PASS,
            1 => preemph(50e-6, fs as f64),
            2 => preemph(75e-6, fs as f64),
            _ => preemph(530e-6, fs as f64),
        };
        let limit = [1.0, 0.9, 0.5][round % 3];
        let (hold, release) = (
            [0.0, 0.0001, 0.02][round % 3],
            [0.0005, 0.01, 0.2][round % 3],
        );
        let scale = [0.1, 1.0, 20.0][(round / 3) % 3];
        let x: Vec<f32> = (0..2000)
            .map(|i| {
                if round % 5 == 4 && i % 97 == 5 {
                    specials[i % specials.len()]
                } else {
                    (0.5 * (0.21 * i as f32).sin() + 0.5 * rng.next()) * scale
                }
            })
            .collect();
        let mut mono = PreemphLimitState::<f32>::new(c, limit, hold, release, rate(fs)).unwrap();
        let mut st = PreemphLimitStereoState::new(c, limit, hold, release, rate(fs)).unwrap();
        let mut y = vec![0.0f32; x.len()];
        let stereo_in: Vec<C32> = x.iter().map(|&v| C32::new(v, v)).collect();
        // Two calls with a limit change in between.
        let cut = 1234;
        Fallback
            .preemph_limit(&mut mono, &x[..cut], &mut y[..cut])
            .unwrap();
        let mut out = run_stereo(&Fallback, &mut st, &stereo_in[..cut]);
        mono.set_limit(limit * 0.5).unwrap();
        st.set_limit(limit * 0.5).unwrap();
        Fallback
            .preemph_limit(&mut mono, &x[cut..], &mut y[cut..])
            .unwrap();
        out.extend(run_stereo(&Fallback, &mut st, &stereo_in[cut..]));
        for (i, (o, m)) in out.iter().zip(&y).enumerate() {
            assert_eq!(o.re.to_bits(), m.to_bits(), "round {round} sample {i}");
            assert_eq!(o.im.to_bits(), m.to_bits(), "round {round} sample {i}");
        }
        assert_eq!(st.gain(), mono.gain());
    }
}

/// A loud left channel lowers the right channel by the same factor: while limiting, the
/// output of each channel is its pre-emphasized value times one common factor.
#[test]
fn the_gain_is_linked() {
    let c = preemph(50e-6, 48_000.0);
    let (b0, b1, a1) = (c[0] as f32, c[1] as f32, c[2] as f32);
    let input = audio(6000, 7, 3.0);
    let mut s = stereo(c, 0.9, 0.02, 0.2);
    let out = run_stereo(&Fallback, &mut s, &input);
    let (mut x1, mut v1) = (C32::new(0.0, 0.0), C32::new(0.0, 0.0));
    let mut limited = 0;
    for (x, o) in input.iter().zip(&out) {
        let v = C32::new(
            (b0 * x.re + b1 * x1.re) - a1 * v1.re,
            (b0 * x.im + b1 * x1.im) - a1 * v1.im,
        );
        x1 = *x;
        v1 = v;
        assert!(o.re.abs() <= 0.9 && o.im.abs() <= 0.9);
        let (gl, gr) = (o.re / v.re, o.im / v.im);
        if v.re.abs() > 1e-3 && v.im.abs() > 1e-3 && o.re.abs() < 0.9 && o.im.abs() < 0.9 {
            assert!((gl - gr).abs() <= 1e-5 * gl.abs(), "{gl} {gr}");
            if gl < 0.999 {
                limited += 1;
            }
        }
    }
    // The right channel was lowered during the bursts on the left.
    assert!(limited > 500, "{limited}");
}

#[test]
fn non_finite_input_counts_as_zero_per_channel() {
    let mut a = stereo(PASS, 1.0, 0.0, 0.2);
    let mut b = stereo(PASS, 1.0, 0.0, 0.2);
    let bad = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let input: Vec<C32> = (0..30)
        .map(|i| C32::new(bad[i % 3], 0.25 * (i as f32 * 0.5).sin()))
        .collect();
    let zeroed: Vec<C32> = input.iter().map(|x| C32::new(0.0, x.im)).collect();
    assert_eq!(
        bits(&run_stereo(&Fallback, &mut a, &input)),
        bits(&run_stereo(&Fallback, &mut b, &zeroed))
    );
    assert_eq!(a, b);
    // Silence stays silence, full scale passes at limit 1.
    let mut s = stereo(PASS, 1.0, 0.02, 0.2);
    let out = run_stereo(&Fallback, &mut s, &[C32::new(0.0, 0.0); 16]);
    assert!(out.iter().all(|o| o.re == 0.0 && o.im == 0.0));
    let out = run_stereo(&Fallback, &mut s, &[C32::new(1.0, -1.0); 4]);
    assert!(out.iter().all(|o| *o == C32::new(1.0, -1.0)));
}

#[test]
fn the_output_never_exceeds_the_limit() {
    let mut rng = Rng(489);
    for (round, &limit) in [1.0, 0.9, 0.3].iter().enumerate() {
        let c = preemph([50e-6, 75e-6, 530e-6][round], 48_000.0);
        let mut s = stereo(c, limit, 0.001, 0.01);
        let input: Vec<C32> = (0..8000)
            .map(|_| C32::new(20.0 * rng.next(), 20.0 * rng.next()))
            .collect();
        let t = limit as f32;
        for o in run_stereo(&Fallback, &mut s, &input) {
            assert!(o.re.abs() <= t && o.im.abs() <= t, "{o}");
        }
    }
}

#[test]
fn the_stereo_limiter_is_block_size_invariant() {
    let input = audio(5000, 3, 2.0);
    let c = preemph(75e-6, 48_000.0);
    let mut whole = stereo(c, 0.8, 0.005, 0.05);
    let expected = run_stereo(&Fallback, &mut whole, &input);
    for chunk in [1usize, 7, 64, 333, 4096] {
        let mut s = stereo(c, 0.8, 0.005, 0.05);
        let mut got = Vec::new();
        for part in input.chunks(chunk) {
            got.extend(run_stereo(&Fallback, &mut s, part));
        }
        assert_eq!(bits(&got), bits(&expected), "chunk {chunk}");
        assert_eq!(s, whole);
    }
}

#[test]
fn stereo_encoder_construction_rules() {
    let r = rate(192_000);
    assert!(StereoEncodeState::new(19_000.0, 0.9, 0.09, r).is_ok());
    assert!(StereoEncodeState::new(18_000.0, 0.0, 0.0, r).is_ok());
    assert!(StereoEncodeState::new(20_000.0, 1.0, 0.2, rate(120_000)).is_ok());
    for f in [17_999.0, 20_001.0, f64::NAN, f64::INFINITY] {
        assert!(invalid(StereoEncodeState::new(f, 0.9, 0.09, r)), "{f}");
    }
    for m in [-0.01, 1.01, f32::NAN] {
        assert!(invalid(StereoEncodeState::new(19_000.0, m, 0.09, r)), "{m}");
    }
    for p in [-0.01, 0.21, f32::NAN, f32::INFINITY] {
        assert!(invalid(StereoEncodeState::new(19_000.0, 0.9, p, r)), "{p}");
    }
    assert!(invalid(StereoEncodeState::new(
        19_000.0,
        0.9,
        0.09,
        rate(119_999)
    )));
    let mut s = StereoEncodeState::new(19_000.0, 0.9, 0.09, r).unwrap();
    assert_eq!(s.increment(), Nco::new(19_000.0, r).unwrap().increment());
    assert_eq!((s.ratio(), s.group_delay()), (Ratio::ONE, Delay::ZERO));
    assert!(invalid(s.set_pilot(0.2001)));
    assert!(invalid(s.set_pilot(f32::NAN)));
    assert_eq!(s.pilot(), 0.09);
    s.set_pilot(0.2).unwrap();
    assert_eq!((s.pilot(), s.main()), (0.2, 0.9));
    let before = s;
    let mut out = [0.0f32; 2];
    assert!(matches!(
        Fallback.stereo_encode(&mut s, &[C32::new(0.0, 0.0); 3], &mut out),
        Err(Error::LengthMismatch {
            expected: 3,
            got: 2
        })
    ));
    assert_eq!(s, before);
    run_encode(&Fallback, &mut s, &[C32::new(0.0, 0.0); 5]);
    assert_eq!(s.phase(), s.increment().wrapping_mul(5));
    s.reset();
    assert_eq!((s.phase(), s.pilot()), (0, 0.2));
}

/// The encoder equals the composition of the M30 kernels bit for bit: `oscillate` with the
/// harmonics 1 and 2 (imaginary parts), then `mul_real` and `add_real` in the order of the
/// contract.
#[test]
fn the_encoder_equals_the_m30_composition() {
    let r = rate(192_000);
    let n = 4000;
    let input: Vec<C32> = (0..n)
        .map(|i| {
            let t = i as f32 / 192_000.0;
            C32::new(
                0.6 * (2.0 * core::f32::consts::PI * 1000.0 * t).sin(),
                0.4 * (2.0 * core::f32::consts::PI * 3000.0 * t).sin(),
            )
        })
        .collect();
    let (main, pilot) = (0.9f32, 0.09f32);
    let mut s = StereoEncodeState::new(19_000.0, main, pilot, r).unwrap();
    s.set_phase(0x4000_0000);
    let got = run_encode(&Fallback, &mut s, &input);

    let mut nco = Nco::new(19_000.0, r).unwrap();
    nco.set_phase(0x4000_0000);
    let (mut h1, mut h2) = (vec![C32::new(0.0, 0.0); n], vec![C32::new(0.0, 0.0); n]);
    let mut nco2 = nco;
    Fallback.oscillate(&mut nco, 1, &mut h1).unwrap();
    Fallback.oscillate(&mut nco2, 2, &mut h2).unwrap();
    let s1: Vec<f32> = h1.iter().map(|v| v.im).collect();
    let s2: Vec<f32> = h2.iter().map(|v| v.im).collect();
    let m: Vec<f32> = input.iter().map(|x| (x.re + x.im) * 0.5).collect();
    let d: Vec<f32> = input.iter().map(|x| (x.re - x.im) * 0.5).collect();
    let mut main_m = vec![0.0f32; n];
    let mut main_d = vec![0.0f32; n];
    let mut diff = vec![0.0f32; n];
    let mut tone = vec![0.0f32; n];
    let mut partial = vec![0.0f32; n];
    let mut expected = vec![0.0f32; n];
    Fallback.mul_real(&vec![main; n], &m, &mut main_m).unwrap();
    Fallback.mul_real(&vec![main; n], &d, &mut main_d).unwrap();
    Fallback.mul_real(&main_d, &s2, &mut diff).unwrap();
    Fallback.mul_real(&vec![pilot; n], &s1, &mut tone).unwrap();
    Fallback.add_real(&main_m, &diff, &mut partial).unwrap();
    Fallback.add_real(&partial, &tone, &mut expected).unwrap();
    assert_eq!(bits_real(&got), bits_real(&expected));
    assert_eq!(s.phase(), nco.phase());
    // Within 1e-6 of the f64 formula with exact sines as well (the table interpolation).
    for (i, (&y, x)) in got.iter().zip(&input).enumerate() {
        let phi = 2.0
            * core::f64::consts::PI
            * f64::from(0x4000_0000u32.wrapping_add(s.increment().wrapping_mul(i as u32)))
            / 4_294_967_296.0;
        let (l, rr) = (f64::from(x.re), f64::from(x.im));
        let want = f64::from(main) * (l + rr) * 0.5
            + f64::from(main) * (l - rr) * 0.5 * (2.0 * phi).sin()
            + f64::from(pilot) * phi.sin();
        assert!((f64::from(y) - want).abs() < 1e-6, "{i}: {y} {want}");
    }
}

#[test]
fn pilot_zero_and_mono_input() {
    let r = rate(192_000);
    let input = audio(3000, 11, 0.0);
    let mut with = StereoEncodeState::new(19_000.0, 0.9, 0.09, r).unwrap();
    let mut without = with;
    without.set_pilot(0.0).unwrap();
    let a = run_encode(&Fallback, &mut with, &input);
    let b = run_encode(&Fallback, &mut without, &input);
    // The difference is the pilot alone: 0.09 sin(phi).
    let mut any = false;
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        let phi =
            2.0 * core::f64::consts::PI * f64::from(r_phase(with.increment(), i)) / 4_294_967_296.0;
        assert!((f64::from(x - y) - 0.09 * phi.sin()).abs() < 1e-6);
        any |= x != y;
    }
    assert!(any);
    // Mono input without pilot is `main m` alone, bit for bit.
    let mono: Vec<C32> = input.iter().map(|x| C32::new(x.re, x.re)).collect();
    let mut s = StereoEncodeState::new(19_000.0, 0.9, 0.0, r).unwrap();
    let y = run_encode(&Fallback, &mut s, &mono);
    for (o, x) in y.iter().zip(&mono) {
        assert_eq!(o.to_bits(), (0.9 * ((x.re + x.im) * 0.5)).to_bits());
    }
    // The peak at full scale stays within main + pilot.
    let full: Vec<C32> = (0..3000)
        .map(|i| C32::new([1.0, -1.0][i % 2], [1.0, -1.0][(i / 3) % 2]))
        .collect();
    let mut s = StereoEncodeState::new(19_000.0, 0.9, 0.09, r).unwrap();
    assert!(
        run_encode(&Fallback, &mut s, &full)
            .iter()
            .all(|y| y.abs() <= 0.99 + 1e-6)
    );
}

fn r_phase(inc: u32, i: usize) -> u32 {
    inc.wrapping_mul(i as u32)
}

#[test]
fn the_encoder_is_block_size_invariant_and_passes_non_finite_input() {
    let r = rate(240_000);
    let input = audio(4000, 5, 0.5);
    let mut whole = StereoEncodeState::new(19_000.0, 0.9, 0.09, r).unwrap();
    let expected = run_encode(&Fallback, &mut whole, &input);
    for chunk in [1usize, 5, 128, 999] {
        let mut s = StereoEncodeState::new(19_000.0, 0.9, 0.09, r).unwrap();
        let mut got = Vec::new();
        for part in input.chunks(chunk) {
            got.extend(run_encode(&Fallback, &mut s, part));
        }
        assert_eq!(bits_real(&got), bits_real(&expected), "chunk {chunk}");
        assert_eq!(s, whole);
    }
    // A NaN affects its own sample only.
    let mut bad = input.clone();
    bad[10] = C32::new(f32::NAN, 0.0);
    let mut s = StereoEncodeState::new(19_000.0, 0.9, 0.09, r).unwrap();
    let y = run_encode(&Fallback, &mut s, &bad);
    assert!(y[10].is_nan());
    for (i, (a, b)) in y.iter().zip(&expected).enumerate() {
        if i != 10 {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

#[test]
fn the_public_types_are_send_and_sync() {
    fn assert_send<T: Send + Sync>() {}
    assert_send::<PreemphLimitStereoState>();
    assert_send::<StereoEncodeState>();
}

#[cfg(feature = "reference")]
mod against_the_reference {
    use super::*;
    use sdrk_core::Reference;

    #[test]
    fn the_stereo_limiter_equals_the_reference() {
        let mut rng = Rng(12);
        let specials = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            1e30,
            -2000.0,
        ];
        for round in 0..120 {
            let fs = [44_100u64, 48_000, 96_000, 192_000][round % 4];
            let tau = [0.0, 50e-6, 75e-6, 530e-6, 1000e-6][(round / 4) % 5];
            let c = if tau == 0.0 {
                PASS
            } else {
                preemph(tau, fs as f64)
            };
            let limit = [1.0, 0.85, 0.5, 0.1][round % 4];
            let (hold, release) = (
                [0.0, 0.0001, 0.02][round % 3],
                [0.0005, 0.01, 0.2][round % 3],
            );
            let scale = [0.05, 1.0, 20.0][(round / 7) % 3];
            let n = 1 + (rng.next().abs() * 3000.0) as usize;
            let input: Vec<C32> = (0..n)
                .map(|i| {
                    if round % 6 == 5 && i % 41 == 3 {
                        C32::new(specials[i % specials.len()], rng.next())
                    } else {
                        C32::new(
                            (0.5 * (0.31 * i as f32).sin() + 0.5 * rng.next()) * scale,
                            rng.next() * scale,
                        )
                    }
                })
                .collect();
            let mut a = PreemphLimitStereoState::new(c, limit, hold, release, rate(fs)).unwrap();
            let mut b = a;
            let cut = n / 2;
            let mut ra = run_stereo(&Fallback, &mut a, &input[..cut]);
            let mut rb = run_stereo(&Reference, &mut b, &input[..cut]);
            a.set_limit(limit * 0.5).unwrap();
            b.set_limit(limit * 0.5).unwrap();
            ra.extend(run_stereo(&Fallback, &mut a, &input[cut..]));
            rb.extend(run_stereo(&Reference, &mut b, &input[cut..]));
            assert_eq!(bits(&ra), bits(&rb), "round {round}");
            assert_eq!(a, b, "round {round}: state");
        }
    }

    #[test]
    fn the_encoder_equals_the_reference() {
        let mut rng = Rng(13);
        for round in 0..40 {
            let fs = [120_000u64, 192_000, 228_000, 240_000, 384_000][round % 5];
            let f = [18_000.0, 19_000.0, 19_000.5, 20_000.0][round % 4];
            let main = [0.9f32, 1.0, 0.0, 0.45][round % 4];
            let pilot = [0.09f32, 0.0, 0.2][round % 3];
            let mut a = StereoEncodeState::new(f, main, pilot, rate(fs)).unwrap();
            a.set_phase(rng.0 as u32);
            let mut b = a;
            let n = 1 + (rng.next().abs() * 3000.0) as usize;
            let input: Vec<C32> = (0..n)
                .map(|i| {
                    if round % 5 == 4 && i % 37 == 0 {
                        C32::new(f32::NAN, f32::INFINITY)
                    } else {
                        C32::new(rng.next() * 2.0, rng.next() * 2.0)
                    }
                })
                .collect();
            let ya = run_encode(&Fallback, &mut a, &input);
            let yb = run_encode(&Reference, &mut b, &input);
            assert_eq!(ya.len(), yb.len());
            for (x, y) in ya.iter().zip(&yb) {
                assert!(x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
            }
            assert_eq!(a, b, "round {round}");
        }
    }
}
