//! Tests of `invariance`, `arena`, `virtual_rx` and the suite runner
//! (milestone tests 5 to 7).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::unnecessary_wraps,
    clippy::ignored_unit_patterns,
    clippy::items_after_statements,
    clippy::result_large_err,
    clippy::needless_pass_by_value,
    clippy::match_wildcard_for_single_variants,
    clippy::single_match_else
)]

use num_complex::Complex64;
use sdrk_core::{C32, Progress};
use sdrk_device::{BlockFlags, Rate, RxStream, SampleIndex, TimeAnchor, TimeSource, formats};
use sdrk_testkit::encode::decode_reference;
use sdrk_testkit::{
    Arena, ChunkPlan, Failure, NoiseSource, Reporter, SumSource, Summary, TestCase, Tier,
    ToneSource, VirtualRx, WriteReporter, check_block_invariance, run_suite,
};

trait Norm {
    fn norm(self) -> f64;
}
impl Norm for Complex64 {
    fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }
}
fn polar(r: f64, a: f64) -> Complex64 {
    Complex64::new(r * a.cos(), r * a.sin())
}

type Step<'a, S> = &'a mut dyn FnMut(&mut S, &[f32], &mut [f32]) -> sdrk_core::Result<Progress>;

fn copy_all(_: &mut (), i: &[f32], o: &mut [f32]) -> sdrk_core::Result<Progress> {
    let n = i.len().min(o.len());
    o[..n].copy_from_slice(&i[..n]);
    Ok(Progress {
        consumed: n,
        produced: n,
    })
}

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 + 1.0).collect()
}

#[test]
fn chunk_plan_is_deterministic_and_bounded() {
    let plan = ChunkPlan::new(7, 10);
    let a: Vec<usize> = plan.sizes().take(1000).collect();
    let b: Vec<usize> = plan.sizes().take(1000).collect();
    assert_eq!(a, b);
    assert!(a.iter().all(|s| (1..=10).contains(s)));
    for size in 1..=10 {
        assert!(a.contains(&size), "size {size} never drawn");
    }
    assert_ne!(
        a,
        ChunkPlan::new(8, 10).sizes().take(1000).collect::<Vec<_>>()
    );
    assert!(ChunkPlan::new(1, 0).sizes().take(20).all(|s| s == 1));
    assert_eq!((plan.seed(), plan.max()), (7, 10));
}

#[test]
fn invariance_passes_for_an_identity_kernel() {
    let input = ramp(500);
    let (mut a, mut b) = (vec![0.0f32; 500], vec![0.0f32; 500]);
    let r = check_block_invariance(
        &input,
        &mut a,
        &mut b,
        ChunkPlan::new(1, 17),
        || (),
        copy_all,
    )
    .unwrap();
    assert_eq!(r.outputs, 500);
    assert_eq!(r.calls_whole, 1);
    assert!(r.calls_chunked > 20);
    assert_eq!(a, b);
}

fn running_sum(acc: &mut f32, i: &[f32], o: &mut [f32]) -> sdrk_core::Result<Progress> {
    let n = i.len().min(o.len());
    for (x, y) in i[..n].iter().zip(&mut o[..n]) {
        *acc += x;
        *y = *acc;
    }
    Ok(Progress {
        consumed: n,
        produced: n,
    })
}

#[test]
fn invariance_passes_for_a_correct_stateful_kernel() {
    let input = ramp(300);
    let (mut a, mut b) = (vec![0.0f32; 300], vec![0.0f32; 300]);
    check_block_invariance(
        &input,
        &mut a,
        &mut b,
        ChunkPlan::new(4, 31),
        || 0.0f32,
        running_sum,
    )
    .unwrap();
    assert_eq!(a[299], (300.0 * 301.0 / 2.0) as f32);
}

#[test]
fn invariance_fails_at_the_right_index_for_a_state_resetting_kernel() {
    let input = ramp(300);
    let (mut a, mut b) = (vec![0.0f32; 300], vec![0.0f32; 300]);
    let plan = ChunkPlan::new(4, 31);
    let first_chunk = plan.sizes().next().unwrap();
    let mut broken = |acc: &mut f32, i: &[f32], o: &mut [f32]| {
        *acc = 0.0; // forgets its state at every call
        running_sum(acc, i, o)
    };
    let step: Step<f32> = &mut broken;
    let err = check_block_invariance(&input, &mut a, &mut b, plan, || 0.0f32, step).unwrap_err();
    assert_eq!(err.msg, "output depends on the block size");
    assert_eq!(
        err.detail.as_str(),
        format!("first mismatch at index {first_chunk}")
    );
}

#[test]
fn invariance_handles_partial_consumption_and_decimation() {
    // Consumes at most 3 samples per call and outputs the first of each pair
    // of consumed samples seen so far (a stateful 2:1 decimator).
    let input = ramp(101);
    let (mut a, mut b) = (vec![0.0f32; 101], vec![0.0f32; 101]);
    let step = |phase: &mut usize, i: &[f32], o: &mut [f32]| {
        let n = i.len().min(3);
        let mut produced = 0;
        for x in &i[..n] {
            if *phase % 2 == 0 {
                o[produced] = *x;
                produced += 1;
            }
            *phase += 1;
        }
        Ok(Progress {
            consumed: n,
            produced,
        })
    };
    let r = check_block_invariance(
        &input,
        &mut a,
        &mut b,
        ChunkPlan::new(3, 8),
        || 0usize,
        step,
    )
    .unwrap();
    assert_eq!(r.outputs, 51);
    assert_eq!(a[..3], [1.0, 3.0, 5.0]);
}

#[test]
fn invariance_detects_count_differences_stalls_and_errors() {
    let input = ramp(40);
    let (mut a, mut b) = (vec![0.0f32; 80], vec![0.0f32; 80]);
    // The number of outputs depends on the chunking: emits one extra sample
    // per call.
    let extra = |_: &mut (), i: &[f32], o: &mut [f32]| {
        let n = i.len();
        o[..n].copy_from_slice(i);
        o[n] = 0.0;
        Ok(Progress {
            consumed: n,
            produced: n + 1,
        })
    };
    let e = check_block_invariance(&input, &mut a, &mut b, ChunkPlan::new(1, 5), || (), extra)
        .unwrap_err();
    assert_eq!(e.msg, "output counts differ");
    // Never consumes anything.
    let stall = |_: &mut (), _: &[f32], _: &mut [f32]| Ok(Progress::default());
    let e = check_block_invariance(&input, &mut a, &mut b, ChunkPlan::new(1, 5), || (), stall)
        .unwrap_err();
    assert_eq!(e.msg, "step made no progress");
    // Claims to consume more than offered.
    let liar = |_: &mut (), i: &[f32], _: &mut [f32]| {
        Ok(Progress {
            consumed: i.len() + 1,
            produced: 0,
        })
    };
    let e = check_block_invariance(&input, &mut a, &mut b, ChunkPlan::new(1, 5), || (), liar)
        .unwrap_err();
    assert_eq!(e.msg, "step reported impossible progress");
    // Kernel error is passed on.
    let bad = |_: &mut (), _: &[f32], _: &mut [f32]| Err(sdrk_core::Error::Internal("boom"));
    let e = check_block_invariance(&input, &mut a, &mut b, ChunkPlan::new(1, 5), || (), bad)
        .unwrap_err();
    assert!(e.detail.as_str().contains("boom"));
    // Output buffer too small for the kernel.
    let (mut small_a, mut small_b) = (vec![0.0f32; 10], vec![0.0f32; 10]);
    let strict = |_: &mut (), i: &[f32], o: &mut [f32]| {
        if o.len() < i.len() {
            return Err(sdrk_core::Error::BufferTooSmall {
                needed: i.len(),
                got: o.len(),
            });
        }
        copy_all(&mut (), i, o)
    };
    assert!(
        check_block_invariance(
            &input,
            &mut small_a,
            &mut small_b,
            ChunkPlan::new(1, 5),
            || (),
            strict
        )
        .is_err()
    );
}

#[test]
fn arena_alignment_exhaustion_and_reset() {
    let mut buf = vec![0u8; 256];
    let mut arena = Arena::new(&mut buf);
    let before = arena.remaining();
    assert_eq!(before, 256);
    {
        let bytes = arena.take::<u8>(3).unwrap();
        bytes.fill(0xAA);
    }
    let c = arena.take::<C32>(4).unwrap();
    assert_eq!(c.as_ptr() as usize % std::mem::align_of::<C32>(), 0);
    assert!(
        c.iter().all(|s| s.re == 0.0 && s.im == 0.0),
        "must be zero-filled"
    );
    assert!(arena.remaining() <= 256 - 3 - 32);
    let e = arena.take::<u8>(1000).unwrap_err();
    assert_eq!(e.msg, "out of arena memory");
    // a failed take changes nothing
    assert!(arena.take::<u8>(arena.remaining()).is_ok());
    assert_eq!(arena.remaining(), 0);
    assert!(arena.take::<u8>(1).is_err());
    assert!(arena.take::<u8>(0).is_ok());
    arena.reset();
    assert_eq!(arena.remaining(), 256);
    assert!(arena.take::<u8>(256 - 8).is_ok());
}

#[test]
fn arena_memory_is_zeroed_again_after_reset_and_overflow_is_an_error() {
    let mut buf = vec![0u8; 64];
    let mut arena = Arena::new(&mut buf);
    arena.take::<u8>(32).unwrap().fill(0xFF);
    arena.reset();
    assert!(arena.take::<u8>(32).unwrap().iter().all(|b| *b == 0));
    assert!(arena.take::<u64>(usize::MAX).is_err());
}

#[test]
fn carver_hands_out_disjoint_slices_and_releases_on_drop() {
    let mut buf = vec![0u8; 128];
    let mut arena = Arena::new(&mut buf);
    {
        let mut c = arena.carver();
        let a = c.take::<u8>(1).unwrap();
        let b = c.take::<C32>(2).unwrap();
        let d = c.take::<u16>(3).unwrap();
        a[0] = 1;
        b[0].re = 2.0;
        d[0] = 3;
        assert_eq!(b.as_ptr() as usize % 4, 0);
        assert_eq!(d.as_ptr() as usize % 2, 0);
        let rest = c.remaining();
        assert!(c.take::<u8>(rest + 1).is_err());
        assert_eq!(c.remaining(), rest, "a failed take must not consume memory");
        assert!(c.take::<u8>(rest).is_ok());
    }
    assert_eq!(arena.remaining(), 128);
}

fn rx_format_cs16() -> sdrk_device::SampleFormat {
    formats::CS16
}

#[test]
fn virtual_rx_chunk_sizes_follow_the_plan() {
    let plan = ChunkPlan::new(5, 40);
    let mut scratch = [Complex64::new(0.0, 0.0); 64];
    let src = ToneSource {
        freq: 0.05,
        amp: 0.5,
        phase: 0.0,
    };
    let mut rx = VirtualRx::new(
        rx_format_cs16(),
        Rate::hz(1000).unwrap(),
        src,
        &mut scratch,
        plan,
        &[],
        None,
    )
    .unwrap();
    assert_eq!(rx.format(), formats::CS16);
    assert_eq!(rx.sample_rate(), Rate::hz(1000).unwrap());
    let mut buf = [0u8; 400];
    let mut start = 0u64;
    for want in plan.sizes().take(30) {
        let (bytes, meta) = rx.read(&mut buf).unwrap();
        assert_eq!(bytes, want * 4);
        assert_eq!(meta.start, SampleIndex(start));
        assert_eq!(meta.flags, BlockFlags::empty());
        start += want as u64;
    }
    assert_eq!(rx.next_index(), start);
}

#[test]
fn virtual_rx_is_limited_by_the_buffers_and_chunking_does_not_change_the_signal() {
    let src = ToneSource {
        freq: 0.0123,
        amp: 0.9,
        phase: 0.4,
    };
    let collect = |max: usize, bufsize: usize| -> Vec<u8> {
        let mut scratch = [Complex64::new(0.0, 0.0); 16];
        let mut rx = VirtualRx::new(
            formats::CS16,
            Rate::hz(8000).unwrap(),
            src,
            &mut scratch,
            ChunkPlan::new(max as u64, max),
            &[],
            None,
        )
        .unwrap();
        let mut buf = vec![0u8; bufsize];
        let mut all = Vec::new();
        while all.len() < 4 * 200 {
            let (n, _) = rx.read(&mut buf).unwrap();
            assert!(n <= bufsize && n <= 16 * 4);
            all.extend_from_slice(&buf[..n]);
        }
        all.truncate(4 * 200);
        all
    };
    assert_eq!(collect(3, 400), collect(100, 40));
    let mut scratch = [Complex64::new(0.0, 0.0); 16];
    let mut rx = VirtualRx::new(
        formats::CS16,
        Rate::hz(8000).unwrap(),
        src,
        &mut scratch,
        ChunkPlan::new(1, 4),
        &[],
        None,
    )
    .unwrap();
    assert!(matches!(
        rx.read(&mut [0u8; 3]),
        Err(sdrk_core::Error::BufferTooSmall { .. })
    ));
}

#[test]
fn virtual_rx_applies_gaps_exactly() {
    let gaps = [(10u64, 5u64), (30, 100)];
    let mut scratch = [Complex64::new(0.0, 0.0); 64];
    let src = ToneSource {
        freq: 0.01,
        amp: 0.5,
        phase: 0.0,
    };
    let mut rx = VirtualRx::new(
        formats::CS16,
        Rate::hz(1000).unwrap(),
        src,
        &mut scratch,
        ChunkPlan::new(2, 8),
        &gaps,
        None,
    )
    .unwrap();
    let mut buf = [0u8; 256];
    let mut emitted: Vec<u64> = Vec::new();
    let mut flagged: Vec<u64> = Vec::new();
    let mut expected_samples = Vec::new();
    while emitted.len() < 60 {
        let (n, meta) = rx.read(&mut buf).unwrap();
        let count = n / 4;
        if meta.flags.contains(BlockFlags::GAP_BEFORE) {
            flagged.push(meta.start.0);
        }
        let mut dec = vec![Complex64::new(0.0, 0.0); count];
        decode_reference(&formats::CS16, &buf[..n], &mut dec).unwrap();
        for (k, s) in dec.iter().enumerate() {
            let idx = meta.start.0 + k as u64;
            emitted.push(idx);
            let phase = 2.0 * std::f64::consts::PI * 0.01 * idx as f64;
            expected_samples.push((*s, polar(0.5, phase)));
        }
    }
    // indices 10..15 and 30..130 are skipped, nothing else
    let want: Vec<u64> = (0..10)
        .chain(15..30)
        .chain(130..)
        .take(emitted.len())
        .collect();
    assert_eq!(emitted, want);
    assert_eq!(flagged, [15, 130]);
    // the samples after a gap are the signal at the absolute index
    for (got, ideal) in expected_samples {
        assert!((got - ideal).norm() < 1e-3);
    }
}

#[test]
fn virtual_rx_first_block_carries_the_anchor() {
    let anchor = TimeAnchor {
        index: SampleIndex(0),
        unix_ns: 1_700_000_000_000_000_000,
        source: TimeSource::Gps,
    };
    let mut scratch = [Complex64::new(0.0, 0.0); 8];
    let src = NoiseSource {
        seed: 1,
        power: 0.01,
    };
    let mut rx = VirtualRx::new(
        formats::CF32,
        Rate::hz(1000).unwrap(),
        src,
        &mut scratch,
        ChunkPlan::new(1, 4),
        &[],
        Some(anchor),
    )
    .unwrap();
    let mut buf = [0u8; 64];
    assert_eq!(rx.read(&mut buf).unwrap().1.anchor, Some(anchor));
    assert_eq!(rx.read(&mut buf).unwrap().1.anchor, None);
}

#[test]
fn virtual_rx_validates_its_configuration() {
    let src = ToneSource {
        freq: 0.01,
        amp: 0.5,
        phase: 0.0,
    };
    let rate = Rate::hz(1000).unwrap();
    let plan = ChunkPlan::new(1, 4);
    let mut s1 = [Complex64::new(0.0, 0.0); 8];
    assert!(
        VirtualRx::new(
            formats::CS16,
            rate,
            src,
            &mut s1,
            plan,
            &[(10, 1), (5, 1)],
            None
        )
        .is_err()
    );
    let mut s2 = [Complex64::new(0.0, 0.0); 8];
    // PACKED10 needs groups of two samples; gaps must respect that
    assert!(VirtualRx::new(formats::PACKED10, rate, src, &mut s2, plan, &[(3, 2)], None).is_err());
    let mut s3 = [Complex64::new(0.0, 0.0); 1];
    assert!(VirtualRx::new(formats::PACKED10, rate, src, &mut s3, plan, &[], None).is_err());
    let mut s4 = [Complex64::new(0.0, 0.0); 8];
    let mut rx = VirtualRx::new(
        formats::PACKED10,
        rate,
        src,
        &mut s4,
        ChunkPlan::new(9, 5),
        &[],
        None,
    )
    .unwrap();
    let mut buf = [0u8; 64];
    for _ in 0..20 {
        let (n, _) = rx.read(&mut buf).unwrap();
        assert!(n > 0 && n % 5 == 0, "whole packing groups only, got {n}");
    }
}

#[test]
fn noise_source_is_a_pure_function_of_the_index() {
    let mut a = NoiseSource {
        seed: 9,
        power: 1.0,
    };
    let mut whole = vec![Complex64::new(0.0, 0.0); 100];
    let mut part = vec![Complex64::new(0.0, 0.0); 30];
    use sdrk_testkit::SignalSource;
    a.fill(0, &mut whole);
    a.fill(70, &mut part);
    assert_eq!(&whole[70..], &part[..]);
}

#[test]
fn sum_source_adds_both_summands_for_any_length() {
    let t1 = ToneSource {
        freq: 0.1,
        amp: 0.5,
        phase: 0.0,
    };
    let t2 = ToneSource {
        freq: -0.2,
        amp: 0.25,
        phase: 1.0,
    };
    let mut sum = SumSource { a: t1, b: t2 };
    use sdrk_testkit::SignalSource;
    let mut got = vec![Complex64::new(0.0, 0.0); 150];
    sum.fill(7, &mut got);
    let (mut x, mut y) = (got.clone(), got.clone());
    t1.clone().fill(7, &mut x);
    t2.clone().fill(7, &mut y);
    for i in 0..150 {
        assert_eq!(got[i], x[i] + y[i]);
    }
}

// ---- suite runner -------------------------------------------------------

fn pass(_: &mut Arena<'_>) -> Result<(), Failure> {
    Ok(())
}
fn fail(_: &mut Arena<'_>) -> Result<(), Failure> {
    Err(Failure::new("", "it broke").detail(format_args!("line one\nline two")))
}
fn uses_arena(arena: &mut Arena<'_>) -> Result<(), Failure> {
    // Every case starts with a fully reset arena.
    assert_eq!(arena.remaining(), 128);
    arena.take::<u8>(100)?;
    Ok(())
}

#[derive(Default)]
struct Log(Vec<String>);
impl Reporter for Log {
    fn suite_started(&mut self, total: usize) {
        self.0.push(format!("start {total}"));
    }
    fn started(&mut self, name: &str) {
        self.0.push(format!("started {name}"));
    }
    fn passed(&mut self, name: &str) {
        self.0.push(format!("passed {name}"));
    }
    fn failed(&mut self, f: &Failure) {
        self.0.push(format!("failed {}", f.case));
    }
    fn finished(&mut self, s: &Summary) {
        self.0
            .push(format!("end {} {} {}", s.passed, s.failed, s.skipped));
    }
}

const CASES: [TestCase; 4] = [
    TestCase {
        name: "a_pass",
        tier: Tier::Quick,
        run: pass,
    },
    TestCase {
        name: "b_fail",
        tier: Tier::Full,
        run: fail,
    },
    TestCase {
        name: "c_arena",
        tier: Tier::Quick,
        run: uses_arena,
    },
    TestCase {
        name: "d_arena",
        tier: Tier::Full,
        run: uses_arena,
    },
];

#[test]
fn run_suite_selects_by_tier_resets_the_arena_and_counts() {
    let mut mem = [0u8; 128];
    let mut arena = Arena::new(&mut mem);
    let mut log = Log::default();
    let s = run_suite(&CASES, Tier::Quick, &mut arena, &mut log);
    assert_eq!(
        s,
        Summary {
            passed: 2,
            failed: 0,
            skipped: 2
        }
    );
    assert_eq!(
        log.0,
        [
            "start 2",
            "started a_pass",
            "passed a_pass",
            "started c_arena",
            "passed c_arena",
            "end 2 0 2"
        ]
    );
    let mut log = Log::default();
    let s = run_suite(&CASES, Tier::Full, &mut arena, &mut log);
    assert_eq!(
        s,
        Summary {
            passed: 3,
            failed: 1,
            skipped: 0
        }
    );
    assert!(
        log.0.contains(&"failed b_fail".to_string()),
        "case name is filled in"
    );
}

#[test]
fn write_reporter_prints_the_protocol() {
    let mut mem = [0u8; 128];
    let mut arena = Arena::new(&mut mem);
    let mut rep = WriteReporter::new(String::new());
    run_suite(&CASES, Tier::Full, &mut arena, &mut rep);
    assert_eq!(
        rep.into_inner(),
        "SDRK-SUITE START 4\n\
         SDRK-CASE PASS a_pass\n\
         SDRK-CASE FAIL b_fail it broke line one line two\n\
         SDRK-CASE PASS c_arena\n\
         SDRK-CASE PASS d_arena\n\
         SDRK-SUITE END passed=3 failed=1\n"
    );
}

#[test]
fn failure_detail_is_truncated_not_panicking() {
    let long = "x".repeat(500);
    let f = Failure::new("c", "m").detail(format_args!("{long}"));
    assert_eq!(f.detail.len(), 160);
    assert_eq!(f.to_string().len(), "c: m ()".len() + 160);
}

#[test]
fn registered_cases_have_unique_names_without_spaces() {
    let cases = sdrk_testkit::all_cases();
    assert_eq!(cases.len(), 68);
    for (i, c) in cases.iter().enumerate() {
        assert!(!c.name.contains(' '));
        assert!(cases[i + 1..].iter().all(|o| o.name != c.name));
    }
    for name in [
        "rng_known_answer",
        "gen_tone_frequency",
        "encode_decode_roundtrip_all_formats",
        "scale_fallback_vs_reference",
        "scale_block_invariance",
        "convert_all_formats_vs_encoder",
        "rotate_c32_vs_reference",
        "rotate_cq15_vs_reference",
        "rotate_block_invariance",
        "fir_c32_vs_reference",
        "fir_cq15_vs_reference",
        "fir_block_invariance",
        "fir_impulse_response",
        "extremes_convert_rotate_fir",
        "fir_decimate_c32_vs_reference",
        "fir_decimate_cq15_vs_reference",
        "fir_decimate_block_invariance",
        "resample_c32_vs_reference",
        "resample_cq15_vs_reference",
        "resample_real_vs_reference",
        "resample_block_invariance",
        "quad_demod_c32_vs_reference",
        "quad_demod_cq15_vs_reference",
        "quad_demod_block_invariance",
        "deemphasis_f32_vs_reference",
        "deemphasis_q15_vs_reference",
        "deemphasis_block_invariance",
        "dc_block_c32_vs_reference",
        "dc_block_cq15_vs_reference",
        "dc_block_block_invariance",
        "window_vs_reference",
        "window_generators",
        "mag_sq_vs_reference",
        "power_db_vs_reference",
        "extremes_rx_kernels",
    ] {
        assert!(cases.iter().any(|c| c.name == name), "{name}");
    }
}

fn assert_send<T: Send>() {}
#[test]
fn public_types_are_send() {
    assert_send::<Arena<'static>>();
    assert_send::<sdrk_testkit::Carver<'static>>();
    assert_send::<sdrk_testkit::Pcg32>();
    assert_send::<ChunkPlan>();
    assert_send::<Failure>();
    assert_send::<TestCase>();
    assert_send::<Summary>();
    assert_send::<VirtualRx<'static, ToneSource>>();
    assert_send::<VirtualRx<'static, SumSource<NoiseSource, ToneSource>>>();
    #[cfg(feature = "std")]
    assert_send::<sdrk_testkit::StdReporter>();
    assert_send::<WriteReporter<String>>();
}
