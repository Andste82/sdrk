//! `PipelineDrift`: validation of the node and the output, the effect of `update` on the
//! queue, and no allocation in `update` (counting global allocator).
//!
//! `unsafe` is needed only for `GlobalAlloc`, which is an unsafe trait.
#![allow(unsafe_code)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::assert_is_empty,
    clippy::unreadable_literal,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::redundant_closure_for_method_calls
)]

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    /// Allocations made by the current thread while counting is on.
    static COUNT: Cell<usize> = const { Cell::new(0) };
    static ON: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

fn note() {
    // `try_with`: the thread-local may already be gone during thread teardown.
    let _ = ON.try_with(|on| {
        if on.get() {
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
        }
    });
}

// SAFETY: every call is forwarded unchanged to the system allocator; the bookkeeping only
// touches thread-local `Cell`s and never allocates.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn count_allocations<R>(f: impl FnOnce() -> R) -> (R, usize) {
    COUNT.with(|c| c.set(0));
    ON.with(|on| on.set(true));
    let r = f();
    ON.with(|on| on.set(false));
    (r, COUNT.with(Cell::get))
}

use sdrk_blocks::DriftConfig;
use sdrk_pipeline::drift::PipelineDrift;
use sdrk_pipeline::{BuildOptions, Error, InputData, Pipeline, PipelineSpec, Registry};

const RATE: f64 = 44_100.0;

fn spec(tunable: bool) -> PipelineSpec {
    let mut spec = PipelineSpec::from_json(&common::flow("resample_audio")).unwrap();
    if tunable {
        spec.nodes
            .get_mut("to_cd")
            .unwrap()
            .params
            .insert("tunable".into(), true.into());
    }
    spec
}

fn build(spec: &PipelineSpec) -> Pipeline {
    Registry::standard()
        .build(spec, &BuildOptions::default())
        .unwrap_or_else(|r| panic!("{}", r.to_json()))
}

fn config() -> DriftConfig {
    DriftConfig::new(RATE, 4_410.0)
}

#[test]
fn accepts_a_tunable_resample_node_on_the_output_path() {
    let p = build(&spec(true));
    assert!(PipelineDrift::new(&p, "audio_out", "to_cd", config()).is_ok());
}

#[test]
fn rejects_a_node_that_is_not_tunable_and_unknown_names() {
    let p = build(&spec(false));
    let Err(Error::Build(report)) = PipelineDrift::new(&p, "audio_out", "to_cd", config()) else {
        panic!("a node without tunable must be refused");
    };
    assert_eq!(report.errors.len(), 1);
    assert_eq!(report.errors[0].path, "nodes.to_cd.params.tunable");

    let p = build(&spec(true));
    assert!(matches!(
        PipelineDrift::new(&p, "nope", "to_cd", config()),
        Err(Error::UnknownOutput(_))
    ));
    assert!(matches!(
        PipelineDrift::new(&p, "audio_out", "nope", config()),
        Err(Error::UnknownNode(_))
    ));
    let mut bad = config();
    bad.max_ppm = 5_000.0;
    assert!(matches!(
        PipelineDrift::new(&p, "audio_out", "to_cd", bad),
        Err(Error::Block { .. })
    ));
}

const TWO_OUTPUTS: &str = r#"{
  "sdrk": 1, "name": "two",
  "settings": { "precision": "f32", "max_chunk": 4096 },
  "inputs": { "a": { "type": "f32", "rate_hz": 48000 } },
  "nodes": {
    "gain": { "kind": "delay", "params": { "samples": 3 } },
    "to_cd": { "kind": "resample", "params": { "rate_out_hz": 44100, "tunable": true } },
    "other": { "kind": "resample", "params": { "rate_out_hz": 44100, "tunable": true } }
  },
  "edges": [
    { "from": "a", "to": "gain" }, { "from": "gain", "to": "to_cd" },
    { "from": "to_cd", "to": "out" }, { "from": "a", "to": "other" },
    { "from": "other", "to": "out2" }
  ],
  "outputs": { "out": { "type": "stream" }, "out2": { "type": "stream" } }
}"#;

#[test]
fn rejects_a_node_of_another_kind_and_a_node_off_the_path() {
    let p = build(&PipelineSpec::from_json(TWO_OUTPUTS).unwrap());
    assert!(PipelineDrift::new(&p, "out", "to_cd", config()).is_ok());
    assert!(PipelineDrift::new(&p, "out2", "other", config()).is_ok());
    for (output, node) in [("out", "gain"), ("out", "other"), ("out2", "to_cd")] {
        let r = PipelineDrift::new(&p, output, node, config());
        assert!(matches!(r, Err(Error::Build(_))), "{output} {node}");
    }
}

#[test]
fn update_reads_the_queue_plus_extra_fill_and_sets_ppm() {
    let mut p = build(&spec(true));
    let mut drift = PipelineDrift::new(&p, "audio_out", "to_cd", config()).unwrap();
    p.push("audio_in", InputData::F32(&[0.0; 4_000])).unwrap();
    let queued = p.available("audio_out").unwrap();
    assert!(queued > 3_000);
    assert_eq!(drift.update(&mut p, 0.0, 0).unwrap(), 0.0);
    // Far above the target: the controller lowers ppm.
    let ppm = drift.update(&mut p, 1.0, 5_000).unwrap();
    assert!(ppm < 0.0, "{ppm}");
    assert_eq!(drift.controller().ppm(), ppm);
    assert_eq!(p.config().nodes["to_cd"].params["ppm"], ppm);
}

#[test]
fn update_does_not_allocate() {
    let mut p = build(&spec(true));
    let mut drift = PipelineDrift::new(&p, "audio_out", "to_cd", config()).unwrap();
    let block = [0.0f32; 480];
    let mut out = [0.0f32; 441];
    p.push("audio_in", InputData::F32(&block)).unwrap();
    drift.update(&mut p, 0.0, 0).unwrap();
    let (changed, allocations) = count_allocations(|| {
        let mut last = 0.0;
        let mut changed = false;
        for i in 1..2000u32 {
            let ppm = drift.update(&mut p, f64::from(i) * 0.01, 100).unwrap();
            changed |= ppm != last;
            last = ppm;
        }
        changed
    });
    assert!(changed, "the controller moved the output");
    assert_eq!(allocations, 0);
    // The node still works after many parameter changes.
    p.push("audio_in", InputData::F32(&block)).unwrap();
    assert!(p.read_stream("audio_out", &mut out).unwrap() > 0);
}

/// M28 test 5: input at 48 kHz * (1 + 50 ppm) in blocks of 480, output read in blocks of 441
/// at 44.1 kHz, `PipelineDrift` after every read, 10 minutes of simulated time.
#[test]
fn keeps_the_queue_between_two_clocks_for_ten_minutes() {
    const IN_PERIOD: f64 = 480.0 / (48_000.0 * (1.0 + 50e-6));
    const OUT_PERIOD: f64 = 441.0 / 44_100.0;
    let mut p = build(&spec(true));
    let mut drift = PipelineDrift::new(&p, "audio_out", "to_cd", config()).unwrap();
    let block = [0.0f32; 480];
    let mut out = [0.0f32; 441];
    // Start about half way to the 100 ms target so that the first second is not starved.
    for _ in 0..10 {
        p.push("audio_in", InputData::F32(&block)).unwrap();
    }
    let (mut next_in, mut next_out) = (IN_PERIOD, OUT_PERIOD);
    let mut reads = 0u64;
    let mut short_reads_after_1s = 0u64;
    let mut worst_queue = 0usize;
    loop {
        if next_in <= next_out {
            // An overflowing output (drop = error) makes `push` fail.
            p.push("audio_in", InputData::F32(&block)).unwrap();
            next_in += IN_PERIOD;
        } else {
            let now = next_out;
            let n = p.read_stream("audio_out", &mut out).unwrap();
            if now > 1.0 && n < out.len() {
                short_reads_after_1s += 1;
            }
            drift.update(&mut p, now, 0).unwrap();
            worst_queue = worst_queue.max(p.available("audio_out").unwrap());
            reads += 1;
            next_out += OUT_PERIOD;
            if now >= 600.0 {
                break;
            }
        }
    }
    assert!(reads > 59_000);
    assert_eq!(short_reads_after_1s, 0);
    assert!(worst_queue < 16_384, "{worst_queue}");
}
