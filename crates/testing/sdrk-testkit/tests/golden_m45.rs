//! Golden and response tests of the M45 kernel `channelize` and the design
//! `channelizer_prototype` (tests 1 and 2 of M45).
//!
//! `channelize`: every case (seeds 660 and 661: 16 and 64 channels, a tone per channel and noise,
//! edges 0.5 / 1.0) runs through the facade `Sdrk`, `Fallback` and `Reference` in one piece and
//! through the facade and `Fallback` in 20 chunk plans (seed 662, chunks of 1 to 700 samples);
//! every run must reach the floor of the contract and the regression guard of the case against
//! the exactly rounded f64 definition, and the chunked runs must give the bits of the whole run.
//! The stored taps must be the Rust prototype rounded to `f32`. The responses: a tone swept across
//! the band in 200 steps (16 channels, 0.5 / 1.0, 8 taps per branch) appears with at most 0.1 dB
//! ripple in the strongest channel and at least 60 dB down in every channel two spacings or more
//! away; with 0.4 / 0.6 and 24 taps per branch a tone swept over +-0.4 spacing of a channel stays
//! within 0.1 dB and a tone 0.6 spacings or more from the centre is at least 60 dB down.
//!
//! `channelizer_prototype`: the golden cases of 0.5 / 1.0 and 0.4 / 0.6 within 1e-12, the tap
//! estimate against `scipy.signal.kaiserord` for 20 configurations (seed 663, equal counts), and
//! invalid parameters.
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
    clippy::many_single_char_names,
    clippy::print_stderr
)]

use sdrk::Sdrk;
use sdrk_blocks::Error as BlockError;
use sdrk_blocks::design::{channelizer_prototype, channelizer_taps, channelizer_taps_per_branch};
use sdrk_core::kernels::channelize::{Channelize, ChannelizeState, history_len};
use sdrk_core::{C32, Fallback, Reference};
use sdrk_testkit::ChunkPlan;
use sdrk_testkit::golden::{self, Array, GoldenCase, Tolerance};

/// The seed of the chunk plans (M45 test 1); plan `k` uses `PLAN_SEED * 100 + k`.
const PLAN_SEED: u64 = 662;
/// Chunk plans per case.
const PLANS: u64 = 20;
/// The largest chunk of the plans.
const MAX_CHUNK: usize = 700;

const KERNEL_CASES: [&str; 2] = ["m16_l8_c32_4096", "m64_l8_c32_4096"];
/// The design cases: name, channels, taps per branch, passband, stopband.
const DESIGN_CASES: [(&str, usize, usize, f64, f64); 2] = [
    ("channelizer_m16_l8_0p5_1p0", 16, 8, 0.5, 1.0),
    ("channelizer_m16_l24_0p4_0p6", 16, 24, 0.4, 0.6),
];

type State = ChannelizeState<C32, Vec<f32>, Vec<C32>, Vec<C32>>;

#[test]
fn the_cases_of_m45_are_present() {
    assert_eq!(golden::list("channelize").unwrap(), KERNEL_CASES);
    let design = golden::list("design").unwrap();
    for (name, ..) in DESIGN_CASES {
        assert!(design.iter().any(|n| n == name), "missing design/{name}");
    }
    assert!(design.iter().any(|n| n == "channelizer_taps_table"));
}

fn usize_param(case: &GoldenCase, name: &str) -> usize {
    usize::try_from(case.param_u64(name).unwrap()).unwrap()
}

fn state(taps: &[f32], m: usize) -> State {
    let l = taps.len() / m;
    ChannelizeState::new(
        taps.to_vec(),
        m,
        vec![C32::default(); history_len(m, l)],
        vec![C32::default(); m / 2],
    )
    .unwrap()
}

/// One run: the whole input in one call, or the chunks of plan `k` with room for every step.
fn run<K: Channelize<C32>>(
    kernel: &K,
    taps: &[f32],
    m: usize,
    input: &[C32],
    plan: Option<u64>,
) -> Vec<C32> {
    let mut st = state(taps, m);
    let d = m / 2;
    let mut out = vec![C32::default(); input.len() / d * m];
    match plan {
        None => {
            let p = kernel.channelize(&mut st, input, &mut out).unwrap();
            assert_eq!((p.consumed, p.produced), (input.len(), out.len()));
        }
        Some(k) => {
            let mut sizes = ChunkPlan::new(PLAN_SEED * 100 + k, MAX_CHUNK).sizes();
            let (mut pos, mut written) = (0, 0);
            while pos < input.len() {
                let a = sizes.next().unwrap().min(input.len() - pos);
                let p = kernel
                    .channelize(&mut st, &input[pos..pos + a], &mut out[written..])
                    .unwrap();
                assert_eq!(p.consumed, a);
                pos += a;
                written += p.produced;
            }
            assert_eq!(written, out.len());
        }
    }
    out
}

fn bits(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

/// The prototype of the edges 0.5 / 1.0 at 60 dB rounded to `f32`.
fn prototype_f32(m: usize, l: usize, passband: f64, stopband: f64) -> Vec<f32> {
    let mut h = vec![0.0f64; m * l];
    channelizer_prototype(m, l, passband, stopband, 60.0, &mut h).unwrap();
    h.iter().map(|&v| v as f32).collect()
}

#[test]
fn channelize_matches_the_golden_cases() {
    for name in KERNEL_CASES {
        let case = golden::load("channelize", name).unwrap();
        let m = usize_param(&case, "channels");
        let l = usize_param(&case, "taps_per_branch");
        let Array::CF32(input) = case.array("input").unwrap() else {
            panic!("input must be cf32");
        };
        let Array::F32(taps) = case.array("taps").unwrap() else {
            panic!("taps must be f32");
        };
        let expected = case.array("expected").unwrap();
        // The stored taps are the Rust prototype of the stored edges, rounded to f32.
        let p = case.param_f64("passband").unwrap();
        let s = case.param_f64("stopband").unwrap();
        assert_eq!(case.param_f64("attenuation_db").unwrap(), 60.0);
        assert_eq!(
            bits_f32(taps),
            bits_f32(&prototype_f32(m, l, p, s)),
            "{name}"
        );
        let whole_facade = run(&Sdrk, taps, m, input, None);
        let whole_fallback = run(&Fallback, taps, m, input, None);
        let reference = run(&Reference, taps, m, input, None);
        for (who, out) in [
            ("facade", &whole_facade),
            ("fallback", &whole_fallback),
            ("reference", &reference),
        ] {
            let c = golden::assert_matches(&case, expected, out)
                .unwrap_or_else(|e| panic!("{name} {who}: {e}"));
            eprintln!("{name} {who}: {:.1} dB", c.snr_db);
        }
        for k in 0..PLANS {
            let f = run(&Fallback, taps, m, input, Some(k));
            assert_eq!(bits(&f), bits(&whole_fallback), "{name} fallback plan {k}");
            let s = run(&Sdrk, taps, m, input, Some(k));
            assert_eq!(bits(&s), bits(&whole_facade), "{name} facade plan {k}");
        }
    }
}

fn bits_f32(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

// --------------------------------------------------------------------------------------------
// Responses: tone sweeps
// --------------------------------------------------------------------------------------------

/// The mean power per channel (in dB) of a unit tone at `f` (in channel spacings from the band
/// centre), after the start-up of the filter.
fn channel_powers<K: Channelize<C32>>(kernel: &K, taps: &[f32], m: usize, f: f64) -> Vec<f64> {
    let n = 4096;
    let input: Vec<C32> = (0..n)
        .map(|i| {
            let cycles = (f / m as f64 * f64::from(i)).rem_euclid(1.0);
            let theta = 2.0 * core::f64::consts::PI * cycles;
            C32::new(theta.cos() as f32, theta.sin() as f32)
        })
        .collect();
    let out = run(kernel, taps, m, &input, None);
    let skip = taps.len() / (m / 2) + 1;
    let steps = out.len() / m;
    let mut p = vec![0.0f64; m];
    for group in out.chunks_exact(m).skip(skip) {
        for (acc, y) in p.iter_mut().zip(group) {
            *acc += f64::from(y.re).powi(2) + f64::from(y.im).powi(2);
        }
    }
    p.iter()
        .map(|v| 10.0 * (v / (steps - skip) as f64).max(1e-30).log10())
        .collect()
}

/// The lane of the channel `c` (signed, in spacings from the centre) in FFT order.
fn lane(c: i64, m: usize) -> usize {
    c.rem_euclid(m as i64) as usize
}

#[test]
fn the_search_grid_is_flat_and_rejects_two_spacings_away() {
    let m = 16;
    let taps = prototype_f32(m, 8, 0.5, 1.0);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut worst = f64::NEG_INFINITY;
    for step in 0..200 {
        // From -M/2 to +M/2 spacings (the whole band).
        let f = -(m as f64) / 2.0 + m as f64 * f64::from(step) / 200.0;
        for (who, p) in [
            ("facade", channel_powers(&Sdrk, &taps, m, f)),
            ("fallback", channel_powers(&Fallback, &taps, m, f)),
        ] {
            let (best, &top) = p
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap();
            lo = lo.min(top);
            hi = hi.max(top);
            for (k, &v) in p.iter().enumerate() {
                let dist = (k as i64 - best as i64).rem_euclid(m as i64);
                let dist = dist.min(m as i64 - dist);
                if dist >= 2 {
                    worst = worst.max(v - top);
                    assert!(
                        v - top <= -60.0,
                        "{who} f {f:.3}: channel {k} at {:.1} dB",
                        v - top
                    );
                }
            }
        }
    }
    eprintln!(
        "search grid: strongest channel {lo:.4} .. {hi:.4} dB, worst rejection {worst:.1} dB"
    );
    assert!(hi - lo <= 0.1, "ripple {:.4} dB", hi - lo);
}

#[test]
fn the_pmr446_plan_is_flat_within_0p4_and_rejects_from_0p6() {
    let m = 16;
    let taps = prototype_f32(m, 24, 0.4, 0.6);
    let channel = 3i64;
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for step in 0..=40 {
        let f = channel as f64 - 0.4 + 0.8 * f64::from(step) / 40.0;
        for p in [
            channel_powers(&Sdrk, &taps, m, f),
            channel_powers(&Fallback, &taps, m, f),
        ] {
            lo = lo.min(p[lane(channel, m)]);
            hi = hi.max(p[lane(channel, m)]);
        }
    }
    eprintln!("pmr446 plan: passband {lo:.4} .. {hi:.4} dB");
    assert!(hi - lo <= 0.1 && lo.abs() <= 0.1, "{lo} .. {hi}");
    let mut worst = f64::NEG_INFINITY;
    // Offsets from 0.6 to 8 spacings on both sides of the channel.
    for step in 0..=148 {
        let off = 0.6 + 0.05 * f64::from(step);
        for f in [channel as f64 + off, channel as f64 - off] {
            for p in [
                channel_powers(&Sdrk, &taps, m, f),
                channel_powers(&Fallback, &taps, m, f),
            ] {
                let v = p[lane(channel, m)];
                worst = worst.max(v);
                assert!(v <= -60.0, "offset {off:.2}: {v:.1} dB");
            }
        }
    }
    eprintln!("pmr446 plan: worst rejection from 0.6 spacings {worst:.1} dB");
}

// --------------------------------------------------------------------------------------------
// channelizer_prototype
// --------------------------------------------------------------------------------------------

fn f64_array(case: &GoldenCase, name: &str) -> Vec<f64> {
    match case.array(name).unwrap() {
        Array::F64(v) => v.clone(),
        other => panic!("{name} must be f64, is {}", other.dtype()),
    }
}

fn i32_array(case: &GoldenCase, name: &str) -> Vec<i32> {
    match case.array(name).unwrap() {
        Array::I32(v) => v.clone(),
        other => panic!("{name} must be i32, is {}", other.dtype()),
    }
}

#[test]
fn channelizer_prototype_matches_scipy() {
    for (name, m, l, p, s) in DESIGN_CASES {
        let case = golden::load("design", name).unwrap();
        assert_eq!(usize_param(&case, "channels"), m);
        assert_eq!(usize_param(&case, "taps_per_branch"), l);
        assert_eq!(case.param_f64("passband").unwrap(), p);
        assert_eq!(case.param_f64("stopband").unwrap(), s);
        let a = case.param_f64("attenuation_db").unwrap();
        let mut h = vec![0.0; m * l];
        channelizer_prototype(m, l, p, s, a, &mut h).unwrap();
        let want = f64_array(&case, "expected");
        let Tolerance::MaxAbsErr { max, .. } = case.tolerance else {
            panic!("design cases use max_abs_err");
        };
        assert_eq!(want.len(), h.len());
        let err = h
            .iter()
            .zip(&want)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(err <= max, "{name}: {err:e}");
    }
}

#[test]
fn the_tap_estimate_equals_kaiserord() {
    let case = golden::load("design", "channelizer_taps_table").unwrap();
    assert!(matches!(case.tolerance, Tolerance::BitExact));
    let channels = i32_array(&case, "channels");
    let passband = f64_array(&case, "passband");
    let stopband = f64_array(&case, "stopband");
    let attenuation = f64_array(&case, "attenuation_db");
    let taps = i32_array(&case, "expected");
    let per_branch = i32_array(&case, "taps_per_branch");
    assert_eq!(channels.len(), 20);
    for i in 0..channels.len() {
        let m = usize::try_from(channels[i]).unwrap();
        let (p, s, a) = (passband[i], stopband[i], attenuation[i]);
        let n = channelizer_taps(m, p, s, a).unwrap();
        assert_eq!(
            n,
            usize::try_from(taps[i]).unwrap(),
            "row {i}: M {m}, {p} / {s}, {a} dB"
        );
        let l = channelizer_taps_per_branch(m, p, s, a).unwrap();
        assert_eq!(l, usize::try_from(per_branch[i]).unwrap(), "row {i}");
    }
    // The examples of M45 section 4 (16 channels, 60 dB).
    assert_eq!(channelizer_taps_per_branch(16, 0.5, 1.0, 60.0).unwrap(), 8);
    assert_eq!(channelizer_taps_per_branch(16, 0.4, 0.6, 60.0).unwrap(), 19);
}

#[test]
fn invalid_edges_and_parameters_are_rejected() {
    let invalid = |r: Result<(), BlockError>, want: &str| match r {
        Err(BlockError::InvalidParameter { name, .. }) => assert_eq!(name, want),
        other => panic!("{want}: {other:?}"),
    };
    let mut h = vec![0.0; 128];
    let design = |m: usize, l: usize, p: f64, s: f64, a: f64| {
        let mut buf = vec![0.0; m.saturating_mul(l).min(1 << 16)];
        channelizer_prototype(m, l, p, s, a, &mut buf)
    };
    for (p, s) in [
        (0.0, 1.0),
        (-0.1, 1.0),
        (f64::NAN, 1.0),
        (f64::INFINITY, 1.0),
    ] {
        invalid(design(16, 8, p, s, 60.0), "passband");
        assert!(channelizer_taps(16, p, s, 60.0).is_err());
    }
    for (p, s) in [
        (0.5, 0.5),
        (0.6, 0.5),
        (0.5, 1.500_000_1),
        (0.5, 2.0),
        (0.5, f64::NAN),
        (0.5, f64::INFINITY),
    ] {
        invalid(design(16, 8, p, s, 60.0), "stopband");
        assert!(matches!(
            channelizer_taps(16, p, s, 60.0),
            Err(BlockError::InvalidParameter {
                name: "stopband",
                ..
            })
        ));
    }
    // The largest edge 1.5 is valid.
    assert!(design(16, 8, 0.5, 1.5, 60.0).is_ok());
    for m in [2usize, 12, 512] {
        invalid(design(m, 8, 0.5, 1.0, 60.0), "channels");
    }
    for l in [3usize, 33] {
        invalid(design(16, l, 0.5, 1.0, 60.0), "taps_per_branch");
    }
    for a in [19.9, 120.1, f64::NAN] {
        assert!(matches!(
            design(16, 8, 0.5, 1.0, a),
            Err(BlockError::OutOfRange {
                name: "attenuation_db",
                ..
            })
        ));
    }
    // A buffer of the wrong length.
    assert!(matches!(
        channelizer_prototype(16, 4, 0.5, 1.0, 60.0, &mut h),
        Err(BlockError::Core(sdrk_core::Error::LengthMismatch {
            expected: 64,
            got: 128
        }))
    ));
}
