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

use sdrk_device::{BlockFlags, BlockMeta, SampleIndex, TimeAnchor, TimeSource};
use sdrk_pipeline::{EventKind, InputData, ParamTime, PullError, PullSource};
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

#[test]
fn metadata_events_timed_parameters_and_pull_do_not_allocate() {
    let bytes = common::cu8_recording(8_000);
    let mut pipeline = common::build(&common::flow("fm_receiver"));
    let mut audio = vec![0f32; 4096];
    let mut frame = vec![0f32; 1024];
    // One block is 2000 samples; every 50th block follows a gap of 777 samples.
    let starts: Vec<u64> = (0..1000u64).map(|i| i * 2000 + (i / 50) * 777).collect();
    // The values are built before the measurement: a `ParamTime` owns a `String`.
    let mut times: Vec<ParamTime> = starts
        .iter()
        .rev()
        .map(|s| ParamTime::InputIndex {
            input: "rf".to_string(),
            index: s + 2500,
        })
        .collect();
    let freqs = [json!(600_000.0), json!(610_000.0), json!(590_000)];
    // Warm up outside the measurement.
    pipeline.push("rf", InputData::Raw(&bytes[..2000])).unwrap();
    pipeline.reset();

    let ((events, produced), allocations) = count_allocations(|| {
        let mut events = 0usize;
        let mut produced = 0usize;
        let mut anchors = 0usize;
        for i in 0..1000usize {
            let start = (i * 997) % (bytes.len() - 4096);
            let meta = BlockMeta {
                start: SampleIndex(starts[i]),
                flags: if i % 3 == 0 {
                    BlockFlags::OVERFLOW.union(BlockFlags::RETUNED)
                } else {
                    BlockFlags::empty()
                },
                anchor: (i % 10 == 0).then_some(TimeAnchor {
                    index: SampleIndex(starts[i]),
                    unix_ns: 1_000_000_000 + i as i64,
                    source: TimeSource::Host,
                }),
                tx_time_ns: None,
            };
            pipeline
                .push_with_meta("rf", InputData::Raw(&bytes[start..start + 4000]), &meta)
                .unwrap();
            let t = times.pop().unwrap();
            pipeline
                .set_param_at("tune", "frequency_hz", freqs[i % 3].clone(), t)
                .unwrap();
            produced += pipeline.read_stream("audio", &mut audio).unwrap();
            while pipeline
                .read_frame("spectrum", &mut frame)
                .unwrap()
                .is_some()
            {}
            produced += pipeline.read_index("audio").unwrap() as usize % 2;
            if let Some(Some(t)) = (i % 10 == 5).then(|| pipeline.time_of("audio", 1).unwrap()) {
                produced += usize::from(t > 0);
            }
            while let Some(e) = pipeline.poll_event() {
                events += 1;
                anchors += usize::from(matches!(e.kind, EventKind::Anchor(_)));
            }
            let _ = pipeline.latency("audio").unwrap();
        }
        assert!(anchors > 100, "{anchors}");
        (events, produced)
    });
    assert!(events > 1000, "events were reported: {events}");
    assert!(produced > 1000, "the pipeline did work: {produced}");
    assert_eq!(
        allocations, 0,
        "allocations in 1000 iterations with metadata"
    );
}

#[test]
fn pull_does_not_allocate() {
    pull_without_allocation("fm_transmitter");
}

#[test]
fn pull_of_the_narrow_band_transmitter_does_not_allocate() {
    pull_without_allocation("nfm_transmitter");
}

fn pull_without_allocation(flow: &str) {
    struct Tone {
        data: Vec<f32>,
        pos: usize,
    }
    impl PullSource for Tone {
        fn provide(&mut self, _input: &str, max: usize) -> Result<InputData<'_>, PullError> {
            let from = self.pos % (self.data.len() - 4800);
            self.pos += max;
            Ok(InputData::F32(&self.data[from..from + max]))
        }
    }
    let data: Vec<f32> = (0..20_000)
        .map(|i| ((i % 48) as f32 - 24.0) / 30.0)
        .collect();
    let mut source = Tone { data, pos: 0 };
    let mut pipeline = common::build(&common::flow(flow));
    let mut out = vec![0u8; 3000];
    pipeline.pull("rf", &mut out, &mut source).unwrap();
    let (total, allocations) = count_allocations(|| {
        let mut total = 0usize;
        for i in 0..1000usize {
            let len = 2 * (1 + (i * 37) % 1500);
            let (n, _) = pipeline.pull("rf", &mut out[..len], &mut source).unwrap();
            total += n;
        }
        total
    });
    assert!(total > 1_000_000, "{total}");
    assert_eq!(allocations, 0, "allocations in 1000 pulls");
}

#[test]
fn burst_cycles_do_not_allocate() {
    struct Tone {
        data: Vec<f32>,
    }
    impl PullSource for Tone {
        fn provide(&mut self, _input: &str, max: usize) -> Result<InputData<'_>, PullError> {
            let n = max.min(self.data.len());
            Ok(InputData::F32(&self.data[..n]))
        }
    }
    let data: Vec<f32> = (0..20_000)
        .map(|i| ((i % 48) as f32 - 24.0) / 30.0)
        .collect();
    let mut source = Tone { data };
    let mut pipeline = common::build(&common::flow("fm_transmitter"));
    let mut out = vec![0u8; 3000];
    pipeline.pull("rf", &mut out, &mut source).unwrap();
    let ((), allocations) = count_allocations(|| {
        for _ in 0..50 {
            pipeline.begin_burst("rf", None).unwrap();
            pipeline.pull("rf", &mut out, &mut source).unwrap();
            pipeline.end_burst("rf").unwrap();
            for _ in 0..4 {
                pipeline.pull("rf", &mut out, &mut source).unwrap();
            }
        }
    });
    assert_eq!(allocations, 0, "allocations in burst cycles");
}

#[test]
fn nodes_with_two_inputs_do_not_allocate() {
    // The proof flow joins two paths in `stereo_pll`; an `add` behind a filter gets an automatic
    // delay node. Neither allocates after the build.
    let doc = json!({
        "sdrk": 1, "name": "joins", "settings": { "max_chunk": 4096 },
        "inputs": { "x": { "type": "c32", "rate_hz": 48000 } },
        "nodes": {
            "lp": { "kind": "fir", "params": {
                "cutoff_hz": 4000, "transition_hz": 2500, "attenuation_db": 30
            } },
            "sum": { "kind": "add" }
        },
        "edges": [
            { "from": "x", "to": "sum", "port": "a" },
            { "from": "x", "to": "lp" },
            { "from": "lp", "to": "sum", "port": "b" },
            { "from": "sum", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream", "drop": "drop_oldest" } }
    });
    let mut pipeline = common::build(&doc.to_string());
    assert!(
        pipeline
            .report()
            .nodes
            .iter()
            .any(|n| n.name == "sum__align")
    );
    let input = vec![sdrk::C32::new(0.25, -0.5); 4096];
    let mut out = vec![sdrk::C32::default(); 8192];
    pipeline.push("x", InputData::C32(&input[..100])).unwrap();
    let (produced, allocations) = count_allocations(|| {
        let mut produced = 0usize;
        for i in 0..1000usize {
            pipeline
                .push("x", InputData::C32(&input[..=(i * 37) % 4000]))
                .unwrap();
            produced += pipeline.read_stream("out", &mut out).unwrap();
        }
        produced
    });
    assert!(produced > 10_000, "the pipeline did work: {produced}");
    assert_eq!(
        allocations, 0,
        "allocations between 1000 pushes after build"
    );
}

#[test]
fn packet_inputs_and_outputs_do_not_allocate() {
    // The proof flow `frame_loop` in push mode, and its encoder in pull mode with an idle
    // source between the packets.
    let mut pipeline = common::build(&common::flow("frame_loop"));
    let packets: Vec<sdrk_pipeline::Packet> = (0..64u8)
        .map(|k| {
            let data = vec![k; usize::from(k) * 3];
            sdrk_pipeline::Packet::new(0, sdrk_pipeline::PacketFlags::NONE, &data).unwrap()
        })
        .collect();
    let mut got = vec![sdrk_pipeline::Packet::default(); 128];
    pipeline.push_packets("msg", &packets[..2]).unwrap();
    pipeline.read_packets("out", &mut got).unwrap();
    let (decoded, allocations) = count_allocations(|| {
        let mut decoded = 0usize;
        for i in 0..1000usize {
            let n = 1 + i % 50;
            let from = i % 10;
            pipeline
                .push_packets("msg", &packets[from..from + n])
                .unwrap();
            decoded += pipeline.read_packets("out", &mut got).unwrap();
            assert_eq!(pipeline.available("out").unwrap(), 0);
        }
        decoded
    });
    assert_eq!(decoded, (0..1000).map(|i| 1 + i % 50).sum::<usize>());
    assert_eq!(allocations, 0, "allocations of the packet flow after build");

    struct Sometimes {
        packets: Vec<sdrk_pipeline::Packet>,
        calls: usize,
    }
    impl PullSource for Sometimes {
        fn provide(&mut self, _input: &str, max: usize) -> Result<InputData<'_>, PullError> {
            self.calls += 1;
            if self.calls.is_multiple_of(3) {
                Ok(InputData::Packets(&[]))
            } else {
                let n = max.min(2);
                Ok(InputData::Packets(&self.packets[..n]))
            }
        }
    }
    let doc = json!({
        "sdrk": 1, "name": "tx",
        "inputs": { "msg": { "type": "packets", "rate_hz": 48000 } },
        "nodes": { "enc": { "kind": "frame_encode", "params": { "idle": "preamble" } } },
        "edges": [ { "from": "msg", "to": "enc" }, { "from": "enc", "to": "tx" } ],
        "outputs": { "tx": { "type": "stream" } }
    });
    let mut tx = common::build(&doc.to_string());
    let mut source = Sometimes {
        packets: packets[..4].to_vec(),
        calls: 0,
    };
    let mut samples = vec![0f32; 3000];
    tx.pull_samples::<f32>("tx", &mut samples, &mut source)
        .unwrap();
    let (pulled, allocations) = count_allocations(|| {
        let mut pulled = 0usize;
        for i in 0..1000usize {
            let want = 1 + (i * 131) % 3000;
            pulled += tx
                .pull_samples::<f32>("tx", &mut samples[..want], &mut source)
                .unwrap()
                .0;
        }
        pulled
    });
    assert_eq!(
        pulled,
        (0..1000).map(|i| 1 + (i * 131) % 3000).sum::<usize>()
    );
    assert_eq!(allocations, 0, "allocations of a packet pull after build");

    // Burst cycles behind the packet input: `end_burst` sends the idle tail through the
    // encoder and maps the end along the path without copying it.
    tx.begin_burst("tx", None).unwrap();
    tx.pull_samples::<f32>("tx", &mut samples, &mut source)
        .unwrap();
    tx.end_burst("tx").unwrap();
    tx.pull_samples::<f32>("tx", &mut samples, &mut source)
        .unwrap();
    let ((), allocations) = count_allocations(|| {
        for _ in 0..50 {
            tx.begin_burst("tx", None).unwrap();
            tx.pull_samples::<f32>("tx", &mut samples, &mut source)
                .unwrap();
            tx.end_burst("tx").unwrap();
            for _ in 0..4 {
                tx.pull_samples::<f32>("tx", &mut samples, &mut source)
                    .unwrap();
            }
        }
    });
    assert_eq!(
        allocations, 0,
        "allocations in burst cycles behind a packet input"
    );
}
