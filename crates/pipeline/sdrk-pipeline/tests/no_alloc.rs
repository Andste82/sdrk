//! No allocation after `build`: a counting global allocator watches 1000 iterations of
//! `push`, `read_stream`, `read_frame` and `set_param`.
//!
//! This is the only place of the crate that needs `unsafe`: `GlobalAlloc` is an unsafe trait.
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

use sdrk_pipeline::InputData;
use serde_json::json;

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

#[test]
fn nothing_allocates_after_build() {
    let bytes = common::cu8_recording(8_000);
    let mut pipeline = common::build(&common::flow("fm_receiver"));
    let mut audio = vec![0f32; 4096];
    let mut frame = vec![0f32; 1024];
    // Pre-built values: cloning a number does not allocate.
    let freqs = [json!(600_000.0), json!(610_000.0), json!(590_000)];

    // Warm up once outside the measurement (first use of lazily initialised std parts).
    pipeline.push("rf", InputData::Raw(&bytes[..2000])).unwrap();

    let (produced, allocations) = count_allocations(|| {
        let mut produced = 0usize;
        let mut frames = 0usize;
        for i in 0..1000usize {
            let start = (i * 997) % (bytes.len() - 4096);
            pipeline
                .push("rf", InputData::Raw(&bytes[start..start + 4000]))
                .unwrap();
            produced += pipeline.read_stream("audio", &mut audio).unwrap();
            while pipeline
                .read_frame("spectrum", &mut frame)
                .unwrap()
                .is_some()
            {
                frames += 1;
            }
            pipeline
                .set_param("tune", "frequency_hz", freqs[i % 3].clone())
                .unwrap();
        }
        assert!(frames > 100);
        produced
    });
    assert!(produced > 10_000, "the pipeline did work: {produced}");
    assert_eq!(
        allocations, 0,
        "allocations between 1000 iterations after build"
    );
}

#[test]
fn the_counter_notices_allocations() {
    // Guards the test above against a counter that never counts.
    let (_, n) = count_allocations(|| std::hint::black_box(vec![1u8; 100]));
    assert!(n >= 1);
}
