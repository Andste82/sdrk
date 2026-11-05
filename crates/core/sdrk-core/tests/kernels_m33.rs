//! Tests of the M33 kernels of `sdrk-core`: `noise_blank` and `auto_notch`. Construction rules,
//! timing, the per-sample rules (delay, blanking span, average, non-finite input, the notch on
//! a pure tone), the agreement of `Fallback` and `Reference` and the thread-safety of the
//! public types.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::too_many_lines
)]

use sdrk_core::kernels::auto_notch::{AutoNotch, AutoNotchState};
use sdrk_core::kernels::noise_blank::{NoiseBlank, NoiseBlankState};
use sdrk_core::{
    C32, CQ15, ComplexSample, Delay, Error, Fallback, Q15, Rate, Ratio, Sample, Timing,
};
use sdrk_device::{ErrorKind, SdrkError};

const FS: u64 = 48_000;

fn rate() -> Rate {
    Rate::hz(FS).unwrap()
}

/// A deterministic generator of uniform values in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32).mul_add(2.0, -1.0)
    }
}

/// A tone of amplitude 0.3 plus noise of amplitude 0.01 (power within a factor of 2).
fn clean_c32(n: usize, seed: u64) -> Vec<C32> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|i| {
            let ph = 2.0 * core::f32::consts::PI * 0.05 * i as f32;
            C32::new(
                0.3 * libm::cosf(ph) + 0.01 * rng.next(),
                0.3 * libm::sinf(ph) + 0.01 * rng.next(),
            )
        })
        .collect()
}

fn to_cq15(x: &[C32]) -> Vec<CQ15> {
    x.iter().map(|&v| CQ15::from_c32(v)).collect()
}

fn nb<T: ComplexSample>(threshold_db: f64, hold_s: f64, d: usize) -> NoiseBlankState<T, Vec<T>> {
    NoiseBlankState::new(threshold_db, 0.01, hold_s, vec![T::ZERO; d], rate()).unwrap()
}

/// Runs the fallback over `input` in the given chunk lengths (cycled).
fn run_chunked<T: ComplexSample>(
    st: &mut NoiseBlankState<T, Vec<T>>,
    input: &[T],
    chunks: &[usize],
) -> Vec<T>
where
    Fallback: NoiseBlank<T>,
{
    let mut out = vec![T::ZERO; input.len()];
    let mut at = 0;
    let mut k = 0;
    while at < input.len() {
        let len = chunks[k % chunks.len()].min(input.len() - at);
        Fallback
            .noise_blank(st, &input[at..at + len], &mut out[at..at + len])
            .unwrap();
        at += len;
        k += 1;
    }
    out
}

fn bits_c32(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
}

#[test]
fn noise_blank_construction_validates_every_parameter() {
    let ok = |t: f64, tau: f64, hold: f64, d: usize| {
        NoiseBlankState::<C32, _>::new(t, tau, hold, vec![C32::ZERO; d], rate())
    };
    assert!(ok(6.0, 0.001, 0.0, 0).is_ok());
    assert!(ok(40.0, 1.0, 0.01, 48).is_ok());
    let bad = [
        (5.9, 0.01, 0.0, 0),
        (40.1, 0.01, 0.0, 0),
        (f64::NAN, 0.01, 0.0, 0),
        (f64::INFINITY, 0.01, 0.0, 0),
        (15.0, 0.000_9, 0.0, 0),
        (15.0, 1.1, 0.0, 0),
        (15.0, f64::NAN, 0.0, 0),
        (15.0, 0.01, -0.001, 0),
        (15.0, 0.01, 0.011, 0),
        (15.0, 0.01, f64::NAN, 0),
        // The lookahead is at most round(0.001 * 48000) = 48 samples.
        (15.0, 0.01, 0.0, 49),
    ];
    for (t, tau, hold, d) in bad {
        let e = ok(t, tau, hold, d).unwrap_err();
        assert!(
            matches!(e, Error::InvalidParameter(_)),
            "{t} {tau} {hold} {d}"
        );
        assert_eq!(e.kind(), ErrorKind::InvalidArgument);
        assert!(
            NoiseBlankState::<CQ15, _>::new(t, tau, hold, vec![CQ15::ZERO; d], rate()).is_err()
        );
    }
    // alpha = 1 - exp(-1 / (fs tau)) above 0.5: at 1 kHz a time constant of 1 ms is one sample.
    let slow = Rate::hz(1000).unwrap();
    let e = NoiseBlankState::<C32, _>::new(15.0, 0.001, 0.0, [C32::ZERO; 0], slow).unwrap_err();
    assert!(matches!(e, Error::InvalidParameter(_)));
    // H and D in samples.
    let st = nb::<C32>(15.0, 0.0002, 10);
    assert_eq!((st.lookahead(), st.hold()), (10, 10));
    assert_eq!(
        (st.threshold_db(), st.tau_s(), st.hold_s()),
        (15.0, 0.01, 0.0002)
    );
}

#[test]
fn noise_blank_set_threshold_validates_and_keeps_the_state() {
    let mut st = nb::<CQ15>(15.0, 0.0, 2);
    let input = to_cq15(&clean_c32(100, 1));
    let mut out = vec![CQ15::ZERO; 100];
    Fallback.noise_blank(&mut st, &input, &mut out).unwrap();
    let p = st.average_power();
    assert!(st.set_threshold_db(5.0).is_err());
    assert_eq!(st.threshold_db(), 15.0);
    st.set_threshold_db(30.0).unwrap();
    assert_eq!(st.threshold_db(), 30.0);
    assert_eq!(st.average_power(), p);
}

#[test]
fn noise_blank_timing_is_one_to_one_with_the_lookahead_as_delay() {
    let c = nb::<C32>(15.0, 0.0, 7);
    let q = nb::<CQ15>(15.0, 0.0, 0);
    assert_eq!(c.ratio(), Ratio::ONE);
    assert_eq!(c.group_delay(), Delay::new(7, 1).unwrap());
    assert_eq!(q.ratio(), Ratio::ONE);
    assert_eq!(q.group_delay(), Delay::ZERO);
}

#[test]
fn a_clean_signal_passes_delayed_by_d_bit_exact() {
    let input = clean_c32(4096, 2);
    for d in [0usize, 1, 10, 48] {
        for chunks in [&[4096usize][..], &[1, 7, 300, 3], &[d.max(1), 5]] {
            let mut st = nb::<C32>(15.0, 0.0002, d);
            let out = run_chunked(&mut st, &input, chunks);
            assert!(out[..d].iter().all(|v| *v == C32::ZERO));
            assert_eq!(bits_c32(&out[d..]), bits_c32(&input[..4096 - d]), "d={d}");
            assert_eq!(st.blanked(), 0);
            let q = to_cq15(&input);
            let mut st = nb::<CQ15>(15.0, 0.0002, d);
            let out = run_chunked(&mut st, &q, chunks);
            assert!(out[..d].iter().all(|v| *v == CQ15::ZERO));
            assert_eq!(&out[d..], &q[..4096 - d], "d={d}");
            assert_eq!(st.blanked(), 0);
        }
    }
}

/// The impulse at input `n` zeroes the outputs of the inputs `n - D` to `n + H`: `D + H + 1`
/// output samples starting at output `n`.
fn check_span<T: ComplexSample>(input: &[T], at: usize, d: usize, hold_s: f64, h: usize)
where
    Fallback: NoiseBlank<T>,
{
    let mut st = nb::<T>(15.0, hold_s, d);
    assert_eq!(st.hold(), h as u64);
    let out = run_chunked(&mut st, input, &[input.len()]);
    for (n, y) in out.iter().enumerate() {
        let blank = (at..=at + d + h).contains(&n);
        if blank {
            assert_eq!(*y, T::ZERO, "n={n}");
        } else if n >= d {
            assert_eq!(*y, input[n - d], "n={n}");
        }
    }
    assert_eq!(st.blanked(), (d + h + 1) as u64);
}

#[test]
fn an_impulse_is_blanked_from_the_oldest_sample_of_the_delay_line() {
    let mut input = clean_c32(2000, 3);
    input[1000] = C32::new(2.0, 0.5);
    // CQ15: the signal 20 dB lower, so that a full-scale impulse exceeds it by 15 dB.
    let q = to_cq15(&{
        let mut x: Vec<C32> = input
            .iter()
            .map(|v| C32::new(v.re * 0.1, v.im * 0.1))
            .collect();
        x[1000] = C32::new(0.99, 0.5);
        x
    });
    // hold 0.0002 s at 48 kHz: H = 10 (9.6 rounded).
    for (d, hold_s, h) in [(10, 0.0002, 10), (0, 0.0, 0), (3, 0.0, 0), (0, 0.0002, 10)] {
        check_span(&input, 1000, d, hold_s, h);
        check_span(&q, 1000, d, hold_s, h);
    }
}

#[test]
fn consecutive_impulses_extend_the_blanking_from_the_last_one() {
    let mut input = clean_c32(2000, 4);
    for n in [1000, 1001, 1002, 1010] {
        input[n] = C32::new(-1.5, 1.5);
    }
    let (d, h) = (4usize, 10usize);
    let mut st = nb::<C32>(15.0, 0.0002, d);
    let out = run_chunked(&mut st, &input, &[3, 64]);
    for (n, y) in out.iter().enumerate().skip(d) {
        if (1000..=1010 + d + h).contains(&n) {
            assert_eq!(*y, C32::ZERO, "n={n}");
        } else {
            assert_eq!(*y, input[n - d], "n={n}");
        }
    }
    assert_eq!(st.blanked(), (1010 + d + h + 1 - 1000) as u64);
}

#[test]
fn the_power_of_an_impulse_does_not_enter_the_average() {
    let input = clean_c32(1000, 5);
    let mut powers = Vec::new();
    for amp in [3.0f32, 10.0, 1000.0, f32::INFINITY, f32::NAN] {
        let mut x = input.clone();
        x[999] = C32::new(amp, 0.0);
        let mut st = nb::<C32>(15.0, 0.0, 0);
        run_chunked(&mut st, &x[..999], &[999]);
        let before = st.average_power() as f32;
        run_chunked(&mut st, &x[999..], &[1]);
        // The average moves towards k p, whatever the impulse power.
        let k = libm::pow(10.0, 1.5) as f32;
        let alpha = (1.0 - libm::exp(-1.0 / (FS as f64 * 0.01))) as f32;
        assert_eq!(
            st.average_power() as f32,
            before + alpha * (k * before - before)
        );
        assert_eq!(st.blanked(), 1);
        powers.push(st.average_power());
    }
    assert!(powers.iter().all(|p| *p == powers[0]));
    // CQ15: impulses of different amplitude leave the same average.
    let q: Vec<CQ15> = input
        .iter()
        .map(|v| CQ15::from_c32(C32::new(v.re * 0.1, v.im * 0.1)))
        .collect();
    let mut avg = Vec::new();
    for amp in [16_000i16, 32_767, -32_768] {
        let mut x = q.clone();
        x[999] = CQ15::new(Q15(amp), Q15(amp));
        let mut st = nb::<CQ15>(15.0, 0.0, 0);
        run_chunked(&mut st, &x, &[1000]);
        assert_eq!(st.blanked(), 1);
        avg.push(st.average_power());
    }
    assert!(avg.iter().all(|p| *p == avg[0]));
}

#[test]
fn non_finite_input_is_an_impulse_and_never_reaches_the_output() {
    let mut input = clean_c32(500, 6);
    input[100] = C32::new(f32::NAN, 0.0);
    input[200] = C32::new(0.0, f32::INFINITY);
    input[300] = C32::new(f32::NEG_INFINITY, f32::NAN);
    input[400] = C32::new(3e38, 3e38);
    for (d, hold_s) in [(0usize, 0.0), (5, 0.0), (5, 0.0002)] {
        let mut st = nb::<C32>(15.0, hold_s, d);
        let out = run_chunked(&mut st, &input, &[17]);
        assert!(out.iter().all(|y| y.re.is_finite() && y.im.is_finite()));
        assert!(st.average_power().is_finite() && st.average_power() > 0.0);
        assert_eq!(out[499], input[499 - d]);
    }
    // Non-finite input at the very start (p = 0) is still blanked.
    let mut st = nb::<C32>(15.0, 0.0, 0);
    let out = run_chunked(
        &mut st,
        &[C32::new(f32::NAN, 1.0), C32::new(0.1, 0.1)],
        &[2],
    );
    assert_eq!(out, [C32::ZERO, C32::new(0.1, 0.1)]);
}

#[test]
fn the_first_sample_starts_the_average_and_a_level_step_recovers() {
    // Silence, then a steady signal: the first non-zero sample starts the average, so
    // nothing is blanked.
    let mut input = vec![C32::ZERO; 300];
    input.extend(clean_c32(1000, 7));
    let mut st = nb::<C32>(15.0, 0.0, 0);
    let out = run_chunked(&mut st, &input, &[1300]);
    assert_eq!(bits_c32(&out), bits_c32(&input));
    let mut st = nb::<CQ15>(15.0, 0.0, 0);
    let q = to_cq15(&input);
    assert_eq!(run_chunked(&mut st, &q, &[1300]), q);
    // A step of +30 dB is blanked at first, but the average follows it (towards k p per
    // impulse sample), so the output passes again after a while.
    let mut input = clean_c32(4000, 8);
    for x in &mut input[1000..] {
        *x = C32::new(x.re * 31.6, x.im * 31.6);
    }
    let mut st = nb::<C32>(15.0, 0.0, 0);
    let out = run_chunked(&mut st, &input, &[4000]);
    assert_eq!(out[1000], C32::ZERO);
    assert_eq!(bits_c32(&out[2000..]), bits_c32(&input[2000..]));
    let q: Vec<CQ15> = input
        .iter()
        .map(|&v| CQ15::from_c32(C32::new(v.re / 10.0, v.im / 10.0)))
        .collect();
    let mut st = nb::<CQ15>(15.0, 0.0, 0);
    let out = run_chunked(&mut st, &q, &[4000]);
    assert_eq!(out[1000], CQ15::ZERO);
    assert_eq!(&out[2000..], &q[2000..]);
}

#[test]
fn reset_returns_to_the_new_state() {
    let mut input = clean_c32(600, 9);
    input[300] = C32::new(5.0, 5.0);
    let mut st = nb::<C32>(15.0, 0.0002, 6);
    let first = run_chunked(&mut st, &input, &[600]);
    assert!(st.blanked() > 0);
    st.reset();
    assert_eq!((st.blanked(), st.average_power()), (0, 0.0));
    assert_eq!(
        bits_c32(&run_chunked(&mut st, &input, &[13])),
        bits_c32(&first)
    );
}

#[test]
fn noise_blank_rejects_unequal_lengths() {
    let mut st = nb::<C32>(15.0, 0.0, 2);
    let mut out = [C32::ZERO; 3];
    let e = Fallback
        .noise_blank(&mut st, &[C32::ZERO; 4], &mut out)
        .unwrap_err();
    assert_eq!(
        e,
        Error::LengthMismatch {
            expected: 4,
            got: 3
        }
    );
}

#[test]
fn auto_notch_construction_validates_every_parameter() {
    assert!(AutoNotchState::new([0.0f32; 8], [0.0f32; 9], 1, 0.0001).is_ok());
    assert!(AutoNotchState::new(vec![0.0f32; 256], vec![0.0f32; 320], 64, 0.1).is_ok());
    let invalid = |taps: usize, delay: usize, mu: f32| {
        let e = AutoNotchState::new(vec![0.0f32; taps], vec![0.0f32; taps + delay], delay, mu)
            .unwrap_err();
        assert!(
            matches!(e, Error::InvalidParameter(_)),
            "{taps} {delay} {mu}"
        );
    };
    invalid(7, 8, 0.01);
    invalid(257, 8, 0.01);
    invalid(32, 0, 0.01);
    invalid(32, 65, 0.01);
    invalid(32, 8, 0.000_09);
    invalid(32, 8, 0.11);
    invalid(32, 8, f32::NAN);
    invalid(32, 8, f32::INFINITY);
    let e = AutoNotchState::new([0.0f32; 32], [0.0f32; 41], 8, 0.01).unwrap_err();
    assert_eq!(
        e,
        Error::LengthMismatch {
            expected: 40,
            got: 41
        }
    );
    assert_eq!(e.kind(), ErrorKind::LengthMismatch);
    let st = AutoNotchState::new([1.0f32; 32], [1.0f32; 40], 8, 0.01).unwrap();
    assert!(st.weights().iter().all(|w| *w == 0.0));
    assert_eq!(st.ratio(), Ratio::ONE);
    assert_eq!(st.group_delay(), Delay::ZERO);
}

fn tone(n: usize, f: f32, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| amp * libm::sinf(2.0 * core::f32::consts::PI * f * i as f32))
        .collect()
}

fn power(x: &[f32]) -> f64 {
    x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len() as f64
}

#[test]
fn the_notch_removes_a_pure_tone() {
    // 1 kHz at 8 kHz, L 32, delay 8, mu 0.01.
    let input = tone(16_384, 1000.0 / 8000.0, 0.5);
    let mut st = AutoNotchState::new([0.0f32; 32], [0.0f32; 40], 8, 0.01).unwrap();
    let mut out = vec![0.0f32; input.len()];
    Fallback.auto_notch(&mut st, &input, &mut out).unwrap();
    let att = 10.0 * libm::log10(power(&input[8192..]) / power(&out[8192..]));
    assert!(att > 40.0, "attenuation {att} dB");
    // Before any adaptation the output is the input (zero weights).
    assert_eq!(out[..9], input[..9]);
}

#[test]
fn the_notch_passes_white_noise_mostly() {
    let mut rng = Lcg(10);
    let input: Vec<f32> = (0..16_384).map(|_| 0.3 * rng.next()).collect();
    let mut st = AutoNotchState::new([0.0f32; 32], [0.0f32; 40], 8, 0.01).unwrap();
    let mut out = vec![0.0f32; input.len()];
    Fallback.auto_notch(&mut st, &input, &mut out).unwrap();
    let change = 10.0 * libm::log10(power(&out[8192..]) / power(&input[8192..]));
    assert!(change.abs() < 1.0, "level change {change} dB");
}

#[test]
fn non_finite_audio_gives_zero_and_keeps_the_weights() {
    let mut input = tone(4000, 0.1, 0.5);
    let mut st = AutoNotchState::new([0.0f32; 16], [0.0f32; 20], 4, 0.05).unwrap();
    let mut out = vec![0.0f32; 4000];
    Fallback
        .auto_notch(&mut st, &input[..2000], &mut out[..2000])
        .unwrap();
    let w: Vec<f32> = st.weights().to_vec();
    input[2000] = f32::NAN;
    input[2001] = f32::INFINITY;
    input[2002] = f32::NEG_INFINITY;
    Fallback
        .auto_notch(&mut st, &input[2000..2003], &mut out[2000..2003])
        .unwrap();
    assert_eq!(&out[2000..2003], &[0.0, 0.0, 0.0]);
    assert_eq!(st.weights(), &w[..]);
    Fallback
        .auto_notch(&mut st, &input[2003..], &mut out[2003..])
        .unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(st.weights().iter().all(|v| v.is_finite()));
}

#[test]
fn set_mu_and_reset() {
    let input = tone(1000, 0.1, 0.5);
    let mut st = AutoNotchState::new([0.0f32; 16], [0.0f32; 20], 4, 0.01).unwrap();
    let mut first = vec![0.0f32; 1000];
    Fallback.auto_notch(&mut st, &input, &mut first).unwrap();
    assert!(st.set_mu(0.5).is_err());
    assert_eq!(st.mu(), 0.01);
    st.reset();
    assert!(st.weights().iter().all(|w| *w == 0.0));
    let mut again = vec![0.0f32; 1000];
    Fallback.auto_notch(&mut st, &input, &mut again).unwrap();
    assert_eq!(first, again);
    st.set_mu(0.1).unwrap();
    assert_eq!(st.mu(), 0.1);
    let e = Fallback
        .auto_notch(&mut st, &input, &mut again[..999])
        .unwrap_err();
    assert_eq!(
        e,
        Error::LengthMismatch {
            expected: 1000,
            got: 999
        }
    );
}

#[test]
fn the_notch_is_independent_of_the_block_split() {
    let mut rng = Lcg(11);
    let input: Vec<f32> = tone(5000, 0.13, 0.4)
        .iter()
        .map(|v| v + 0.05 * rng.next())
        .collect();
    let run = |chunks: &[usize]| {
        let mut st = AutoNotchState::new(vec![0.0f32; 24], vec![0.0f32; 31], 7, 0.02).unwrap();
        let mut out = vec![0.0f32; input.len()];
        let (mut at, mut k) = (0, 0);
        while at < input.len() {
            let len = chunks[k % chunks.len()].min(input.len() - at);
            Fallback
                .auto_notch(&mut st, &input[at..at + len], &mut out[at..at + len])
                .unwrap();
            at += len;
            k += 1;
        }
        out.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    };
    let whole = run(&[5000]);
    for chunks in [&[1usize][..], &[7, 31, 1], &[30, 2]] {
        assert_eq!(run(chunks), whole);
    }
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<NoiseBlankState<C32, [C32; 4]>>();
    assert_send::<NoiseBlankState<CQ15, &mut [CQ15]>>();
    assert_sync::<NoiseBlankState<C32, Vec<C32>>>();
    assert_send::<AutoNotchState<[f32; 8], [f32; 9]>>();
    assert_send::<AutoNotchState<&mut [f32], Vec<f32>>>();
    assert_sync::<AutoNotchState<Vec<f32>, Vec<f32>>>();
}

#[cfg(feature = "reference")]
mod against_reference {
    use super::*;
    use sdrk_core::Reference;

    /// Noise and a tone with impulses, NaN and infinity, silence and a level step.
    fn hard_input(n: usize, seed: u64) -> Vec<C32> {
        let mut rng = Lcg(seed);
        let mut x = clean_c32(n, seed);
        for v in &mut x[n / 4..n / 4 + 200] {
            *v = C32::ZERO;
        }
        for v in &mut x[n / 2..] {
            *v = C32::new(v.re * 8.0, v.im * 8.0);
        }
        for i in 0..n / 50 {
            let at = (rng.next().midpoint(1.0) * n as f32) as usize % n;
            x[at] = C32::new(4.0 * rng.next(), 4.0 * rng.next());
            if i % 7 == 0 {
                x[(at + 1) % n] = C32::new(f32::NAN, 0.0);
            }
            if i % 11 == 0 {
                x[(at + 2) % n] = C32::new(f32::INFINITY, -1.0);
            }
        }
        x
    }

    #[test]
    fn noise_blank_fallback_equals_reference() {
        let x = hard_input(20_000, 12);
        let q: Vec<CQ15> = x
            .iter()
            .map(|v| CQ15::from_c32(C32::new(v.re * 0.12, v.im * 0.12)))
            .collect();
        for (t, hold_s, d) in [
            (15.0, 0.0002, 10),
            (6.0, 0.0, 0),
            (40.0, 0.01, 48),
            (10.0, 0.0, 1),
        ] {
            let mut a = nb::<C32>(t, hold_s, d);
            let mut b = a.clone();
            let ya = run_chunked(&mut a, &x, &[20_000]);
            let mut yb = vec![C32::ZERO; x.len()];
            for (i, o) in x.chunks(333).zip(yb.chunks_mut(333)) {
                Reference.noise_blank(&mut b, i, o).unwrap();
            }
            assert_eq!(bits_c32(&ya), bits_c32(&yb), "{t} {hold_s} {d}");
            assert_eq!(a.blanked(), b.blanked());
            assert!(a.blanked() > 0);
            assert_eq!(a.average_power().to_bits(), b.average_power().to_bits());
            let mut a = nb::<CQ15>(t, hold_s, d);
            let mut b = a.clone();
            let ya = run_chunked(&mut a, &q, &[1, 5, 4096]);
            let mut yb = vec![CQ15::ZERO; q.len()];
            Reference.noise_blank(&mut b, &q, &mut yb).unwrap();
            assert_eq!(ya, yb, "{t} {hold_s} {d}");
            assert_eq!(a.blanked(), b.blanked());
            assert!(t >= 40.0 || a.blanked() > 0);
            assert_eq!(a.average_power(), b.average_power());
        }
    }

    #[test]
    fn noise_blank_q15_corners_equal_the_reference() {
        let corners = [-32_768i16, -32_767, -1, 0, 1, 32_767];
        let mut x = Vec::new();
        for &a in &corners {
            for &b in &corners {
                x.extend(std::iter::repeat_n(CQ15::new(Q15(a), Q15(b)), 37));
            }
        }
        let mut a = nb::<CQ15>(6.0, 0.0002, 3);
        let mut b = a.clone();
        let ya = run_chunked(&mut a, &x, &[x.len()]);
        let mut yb = vec![CQ15::ZERO; x.len()];
        Reference.noise_blank(&mut b, &x, &mut yb).unwrap();
        assert_eq!(ya, yb);
    }

    #[test]
    fn auto_notch_fallback_equals_reference() {
        let mut rng = Lcg(13);
        let mut x: Vec<f32> = tone(12_000, 0.11, 0.5)
            .iter()
            .zip(tone(12_000, 0.21, 0.3))
            .map(|(a, b)| a + b + 0.1 * rng.next())
            .collect();
        x[5000] = f32::NAN;
        x[7000] = f32::INFINITY;
        x[7001] = -0.0;
        for (taps, delay, mu) in [
            (32usize, 8usize, 0.01f32),
            (8, 1, 0.1),
            (256, 64, 0.0001),
            (13, 60, 0.05),
        ] {
            let mut a =
                AutoNotchState::new(vec![0.0f32; taps], vec![0.0f32; taps + delay], delay, mu)
                    .unwrap();
            let mut b = a.clone();
            let mut ya = vec![0.0f32; x.len()];
            let mut yb = vec![0.0f32; x.len()];
            Fallback.auto_notch(&mut a, &x, &mut ya).unwrap();
            for (i, o) in x.chunks(97).zip(yb.chunks_mut(97)) {
                Reference.auto_notch(&mut b, i, o).unwrap();
            }
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&ya), bits(&yb), "{taps} {delay} {mu}");
            assert_eq!(bits(a.weights()), bits(b.weights()));
        }
    }
}
