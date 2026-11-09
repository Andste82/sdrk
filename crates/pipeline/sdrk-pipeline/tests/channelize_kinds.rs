//! The kinds of M45: `channelize` and `channel_select` (spec test 6 and the pipeline parts of
//! tests 4 and 5).
//!
//! A `channelize` node with four `channel_select` readers builds and validates, the readers
//! share one lane buffer, their outputs equal the blocks run outside the pipeline, the type
//! checks of lane edges with the real kinds, the report entry of `chan` (`"lanes": 16`, the step
//! rate), no allocation after build (a counting global allocator, as in `no_alloc.rs`), chunk
//! invariance of the whole graph with 20 plans (seed 666), `rate_unsupported` naming the
//! required `taps_per_branch` (3 cases) and the range of `channel`.
//!
//! The global allocator makes `unsafe` necessary: `GlobalAlloc` is an unsafe trait.
#![allow(unsafe_code)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_pass_by_value,
    clippy::unreadable_literal,
    clippy::large_stack_frames
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use sdrk::{C32, Delay, Rate};
use sdrk_blocks::design::channelizer_taps_per_branch;
use sdrk_blocks::{
    Block, ChannelSelectBlock, ChannelSelectConfig, ChannelizeBlock, ChannelizeConfig, LanesMut,
    LanesRef,
};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Pipeline, Registry, Report};
use sdrk_testkit::{ChunkPlan, Pcg32, VecStorage};
use serde_json::{Value, json};

const SEED: u64 = 666;

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

/// The channels of the four readers (signed; -1 twice).
const CHANNELS: [i64; 4] = [0, -1, 7, -1];

/// A `channelize` node `chan` (16 channels, the defaults) with a `channel_select` reader per
/// entry of `channels`, each into its own stream output.
fn flow(channels: &[i64]) -> Value {
    let mut nodes = serde_json::Map::new();
    nodes.insert(
        "chan".into(),
        json!({ "kind": "channelize", "params": { "channels": 16 } }),
    );
    let mut edges = vec![json!({ "from": "iq", "to": "chan" })];
    let mut outputs = serde_json::Map::new();
    for (i, channel) in channels.iter().enumerate() {
        let sel = format!("sel{i}");
        let out = format!("out{i}");
        nodes.insert(
            sel.clone(),
            json!({ "kind": "channel_select", "params": { "channel": channel } }),
        );
        edges.push(json!({ "from": "chan", "to": sel }));
        edges.push(json!({ "from": sel, "to": out }));
        outputs.insert(out, json!({ "type": "stream" }));
    }
    json!({
        "sdrk": 1, "name": "channels",
        "inputs": { "iq": { "type": "c32", "rate_hz": 200000 } },
        "nodes": nodes,
        "edges": edges,
        "outputs": outputs
    })
}

fn validate(spec: &Value) -> Report {
    Registry::standard().validate_json(&spec.to_string(), &BuildOptions::default())
}

fn build(spec: &Value) -> Pipeline {
    Registry::standard()
        .build_json(&spec.to_string(), &BuildOptions::default())
        .unwrap()
}

fn signal(n: usize) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED, 1);
    (0..n)
        .map(|i| {
            // Noise plus a tone one channel spacing below the centre.
            let p = -std::f64::consts::TAU * i as f64 / 16.0;
            C32::new(
                (p.cos() + 0.3 * rng.normal()) as f32,
                (p.sin() + 0.3 * rng.normal()) as f32,
            )
        })
        .collect()
}

/// The output of each reader, computed with the blocks outside the pipeline.
fn reference(x: &[C32], channels: &[i64]) -> Vec<Vec<C32>> {
    let mut chan = ChannelizeBlock::new(&ChannelizeConfig::default(), &mut VecStorage).unwrap();
    let mut lanes = vec![C32::default(); x.len() / 8 * 16];
    let p = chan
        .process(x, LanesMut::new(&mut lanes, 16).unwrap())
        .unwrap();
    lanes.truncate(p.produced * 16);
    channels
        .iter()
        .map(|&channel| {
            let mut sel = ChannelSelectBlock::new(&ChannelSelectConfig { channel }, 16).unwrap();
            let mut out = vec![C32::default(); p.produced];
            sel.process(LanesRef::new(&lanes, 16).unwrap(), &mut out)
                .unwrap();
            out
        })
        .collect()
}

/// Pushes `x` in the chunks of `sizes` and reads every output after each push.
fn run(p: &mut Pipeline, x: &[C32], sizes: &mut dyn Iterator<Item = usize>) -> Vec<Vec<C32>> {
    let names: Vec<String> = (0..CHANNELS.len()).map(|i| format!("out{i}")).collect();
    let mut got = vec![Vec::new(); CHANNELS.len()];
    let mut buf = vec![C32::default(); 8192];
    let mut at = 0;
    while at < x.len() {
        let n = sizes.next().unwrap().min(x.len() - at);
        p.push("iq", InputData::C32(&x[at..at + n])).unwrap();
        at += n;
        for (g, name) in got.iter_mut().zip(&names) {
            let k = p.read_stream(name, &mut buf).unwrap();
            g.extend_from_slice(&buf[..k]);
        }
    }
    got
}

#[test]
fn the_kinds_are_registered() {
    let registry = Registry::standard();
    let chan = registry.factory("channelize").unwrap();
    assert_eq!(chan.kernels(), ["channelize"]);
    let sel = registry.factory("channel_select").unwrap();
    assert_eq!(sel.kernels().len(), 0);
    assert_eq!(registry.descriptions().count(), 69);
}

#[test]
fn four_readers_build_and_get_the_channels_of_the_blocks() {
    let spec = flow(&CHANNELS);
    let report = validate(&spec);
    assert!(report.is_ok(), "{}", report.to_json());
    let mut p = build(&spec);
    let x = signal(8000);
    let got = run(&mut p, &x, &mut std::iter::repeat(1000));
    let want = reference(&x, &CHANNELS);
    assert_eq!(want[0].len(), 1000);
    assert_eq!(got, want);
    // The tone sits in channel -1 (both readers of it), not in the centre channel.
    let power = |v: &[C32]| v[100..].iter().map(C32::norm_sqr).sum::<f32>() / 900.0;
    assert!(power(&got[1]) > 0.9, "{}", power(&got[1]));
    assert!(power(&got[0]) < 0.2, "{}", power(&got[0]));
    assert_eq!(got[1], got[3]);
}

#[test]
fn the_readers_share_one_lane_buffer() {
    let one = validate(&flow(&CHANNELS[..1]));
    let four = validate(&flow(&CHANNELS));
    assert!(one.is_ok() && four.is_ok());
    // One node delivers lanes, whatever the number of readers.
    let lane_nodes = |r: &Report| r.nodes.iter().filter(|n| n.lanes.is_some()).count();
    assert_eq!((lane_nodes(&one), lane_nodes(&four)), (1, 1));
    // Every further reader adds the same bytes (its own output buffer), no lane buffer of its
    // own. That the four readers read one runner buffer is checked inside the crate
    // (`lanes_test.rs`, `the_channelize_kind_has_one_lane_buffer_for_four_readers`).
    let two = validate(&flow(&CHANNELS[..2]));
    let reader = two.memory_bytes - one.memory_bytes;
    assert_eq!(four.memory_bytes - one.memory_bytes, 3 * reader);
    let p = build(&flow(&CHANNELS[..1]));
    assert_eq!(p.report().memory_bytes, one.memory_bytes);
}

#[test]
fn the_report_names_the_width_and_the_step_rate() {
    let report = validate(&flow(&CHANNELS));
    let chan = report.nodes.iter().find(|n| n.name == "chan").unwrap();
    assert_eq!(chan.kind, "channelize");
    assert_eq!(chan.lanes, Some(16));
    assert_eq!(chan.rate_in, Rate::hz(200_000).unwrap());
    assert_eq!(chan.rate_out, Rate::hz(25_000).unwrap());
    // The delay (M L - 1) / 2 input samples.
    assert_eq!(chan.delay_samples_in, Delay::new(127, 2).unwrap());
    let sel = report.nodes.iter().find(|n| n.name == "sel1").unwrap();
    assert_eq!(sel.lanes, None);
    assert_eq!(
        (sel.rate_in, sel.rate_out),
        (Rate::hz(25_000).unwrap(), Rate::hz(25_000).unwrap())
    );
    assert_eq!(sel.clock_domain, "iq");
    // Cost: L + log2(M) 5/2 = 8 + 10 per input sample, plus 1 per step for each reader.
    let want = 18.0 + 4.0 / 8.0;
    assert!(
        (report.cost_per_input_sample - want).abs() < 1e-9,
        "{}",
        report.cost_per_input_sample
    );
    let json: Value = serde_json::from_str(&report.to_json()).unwrap();
    let nodes = json["nodes"].as_array().unwrap();
    let entry = nodes.iter().find(|n| n["name"] == "chan").unwrap();
    assert_eq!(entry["lanes"], 16);
    assert_eq!(entry["rate_out"], 25_000);
    let sel = nodes.iter().find(|n| n["name"] == "sel0").unwrap();
    assert!(sel.get("lanes").is_none());
}

fn errors_of(spec: &Value) -> Vec<sdrk_pipeline::Diagnostic> {
    validate(spec).errors
}

#[test]
fn a_channelizer_into_a_stream_block_is_a_type_mismatch() {
    let spec = json!({
        "sdrk": 1, "name": "bad",
        "inputs": { "iq": { "type": "c32", "rate_hz": 200000 } },
        "nodes": {
            "chan": { "kind": "channelize" },
            "demod": { "kind": "quad_demod" }
        },
        "edges": [
            { "from": "iq", "to": "chan" },
            { "from": "chan", "to": "demod" },
            { "from": "demod", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    let errors = errors_of(&spec);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].code, DiagCode::TypeMismatch);
    assert_eq!(errors[0].path, "edges[1].to");
    assert!(
        errors[0].message.contains("lanes of c32, width 16"),
        "{}",
        errors[0].message
    );
}

#[test]
fn a_stream_into_channel_select_is_a_type_mismatch() {
    let spec = json!({
        "sdrk": 1, "name": "bad",
        "inputs": { "iq": { "type": "c32", "rate_hz": 200000 } },
        "nodes": {
            "dec": { "kind": "ddc", "params": { "decimation": 2 } },
            "sel": { "kind": "channel_select", "params": { "channel": 0 } }
        },
        "edges": [
            { "from": "iq", "to": "dec" },
            { "from": "dec", "to": "sel" },
            { "from": "sel", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    let errors = errors_of(&spec);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].code, DiagCode::TypeMismatch);
    assert_eq!(errors[0].path, "edges[1].to");
    assert!(
        errors[0].message.contains("lanes of c32"),
        "{}",
        errors[0].message
    );
}

#[test]
fn a_channelizer_into_a_stream_output_is_a_type_mismatch() {
    let spec = json!({
        "sdrk": 1, "name": "bad",
        "inputs": { "iq": { "type": "c32", "rate_hz": 200000 } },
        "nodes": { "chan": { "kind": "channelize" } },
        "edges": [ { "from": "iq", "to": "chan" }, { "from": "chan", "to": "all" } ],
        "outputs": { "all": { "type": "stream" } }
    });
    let errors = errors_of(&spec);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].code, DiagCode::TypeMismatch);
    assert_eq!(errors[0].path, "outputs.all.type");
    assert!(
        errors[0].message.contains("lanes of c32, width 16"),
        "{}",
        errors[0].message
    );
}

#[test]
fn channelize_takes_c32_only() {
    for (ty, precision) in [("cq15", "q15"), ("f32", "f32")] {
        let spec = json!({
            "sdrk": 1, "name": "bad", "settings": { "precision": precision },
            "inputs": { "iq": { "type": ty, "rate_hz": 200000 } },
            "nodes": {
                "chan": { "kind": "channelize" },
                "sel": { "kind": "channel_select", "params": { "channel": 0 } }
            },
            "edges": [
                { "from": "iq", "to": "chan" },
                { "from": "chan", "to": "sel" },
                { "from": "sel", "to": "out" }
            ],
            "outputs": { "out": { "type": "stream" } }
        });
        let errors = errors_of(&spec);
        assert!(
            errors.iter().any(|d| d.code == DiagCode::TypeMismatch),
            "{ty}: {errors:?}"
        );
    }
}

/// The diagnostics of one `channelize` node with `params` and a reader of channel 0.
fn channelize_errors(params: Value) -> Vec<sdrk_pipeline::Diagnostic> {
    let mut spec = flow(&[0]);
    spec["nodes"]["chan"]["params"] = params;
    errors_of(&spec)
}

#[test]
fn rate_unsupported_names_the_required_taps_per_branch() {
    let cases = [
        // The PMR446 plan with the default 8 taps per branch needs 19.
        json!({ "channels": 16, "passband": 0.4, "stopband": 0.6 }),
        // 64 sharp channels at 80 dB with 12 taps per branch.
        json!({ "channels": 64, "passband": 0.45, "stopband": 0.8,
                "attenuation_db": 80.0, "taps_per_branch": 12 }),
        // A transition band so narrow that no taps_per_branch up to 32 is enough.
        json!({ "channels": 16, "passband": 0.48, "stopband": 0.52, "taps_per_branch": 32 }),
    ];
    for params in cases {
        let get = |k: &str, d: f64| params.get(k).and_then(Value::as_f64).unwrap_or(d);
        let need = channelizer_taps_per_branch(
            get("channels", 16.0) as usize,
            get("passband", 0.5),
            get("stopband", 1.0),
            get("attenuation_db", 60.0),
        )
        .unwrap();
        let errors = channelize_errors(params.clone());
        assert_eq!(errors.len(), 1, "{params}: {errors:?}");
        let e = &errors[0];
        assert_eq!(e.code, DiagCode::RateUnsupported);
        assert_eq!(e.path, "nodes.chan.params.taps_per_branch");
        assert!(
            e.message.contains(&format!("taps_per_branch {need}")),
            "{params}: {}",
            e.message
        );
        if need > 32 {
            assert!(
                e.message.contains("more than the largest 32"),
                "{}",
                e.message
            );
        }
    }
    // With the required taps per branch the PMR446 plan validates.
    assert_eq!(
        channelize_errors(json!({ "passband": 0.4, "stopband": 0.6, "taps_per_branch": 19 })),
        Vec::new()
    );
}

#[test]
fn bad_channel_counts_and_edges_are_invalid_values() {
    let errors = channelize_errors(json!({ "channels": 12 }));
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].code, DiagCode::InvalidValue);
    assert_eq!(errors[0].path, "nodes.chan.params.channels");
    let errors = channelize_errors(json!({ "passband": 0.8, "stopband": 0.6 }));
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].code, DiagCode::InvalidValue);
    assert_eq!(errors[0].path, "nodes.chan.params.stopband");
    let errors = channelize_errors(json!({ "stopband": 1.6 }));
    assert_eq!(errors[0].code, DiagCode::OutOfRange, "{errors:?}");
}

#[test]
fn a_channel_outside_the_width_names_the_width_and_the_range() {
    // 200 and -129 lie outside -128 .. 127 (the range of the widest input) and still get the
    // message of the width.
    for (channel, ok) in [
        (-8i64, true),
        (7, true),
        (8, false),
        (-9, false),
        (200, false),
        (-129, false),
        (i64::MAX, false),
        (i64::MIN, false),
    ] {
        let errors = errors_of(&flow(&[channel]));
        if ok {
            assert_eq!(errors, Vec::new(), "{channel}");
            continue;
        }
        assert_eq!(errors.len(), 1, "{errors:?}");
        let e = &errors[0];
        assert_eq!(e.code, DiagCode::RateUnsupported);
        assert_eq!(e.path, "nodes.sel0.params.channel");
        assert!(
            e.message.contains("-8 to 7") && e.message.contains("16 lanes"),
            "{}",
            e.message
        );
    }
    // With 32 channels the same channel is valid.
    let mut spec = flow(&[8]);
    spec["nodes"]["chan"]["params"]["channels"] = json!(32);
    assert_eq!(errors_of(&spec), Vec::new());
}

#[test]
fn nothing_allocates_after_build() {
    let mut p = build(&flow(&CHANNELS));
    let x = signal(4096);
    let mut buf = vec![C32::default(); 4096];
    // Warm up once outside the measurement.
    p.push("iq", InputData::C32(&x[..1000])).unwrap();
    let names: Vec<String> = (0..CHANNELS.len()).map(|i| format!("out{i}")).collect();
    let (produced, allocations) = count_allocations(|| {
        let mut produced = 0usize;
        for i in 0..1000usize {
            let start = (i * 997) % (x.len() - 1100);
            p.push("iq", InputData::C32(&x[start..start + 1000 + i % 97]))
                .unwrap();
            for name in &names {
                produced += p.read_stream(name, &mut buf).unwrap();
            }
        }
        produced
    });
    assert!(produced > 400_000, "the pipeline did work: {produced}");
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
fn the_whole_graph_is_chunk_invariant_with_20_plans() {
    let x = signal(6007);
    let want = run(
        &mut build(&flow(&CHANNELS)),
        &x,
        &mut std::iter::repeat(4096),
    );
    assert_eq!(want, reference(&x, &CHANNELS));
    for k in 0..20 {
        let plan = ChunkPlan::new(SEED, 1 + 200 * k);
        let got = run(&mut build(&flow(&CHANNELS)), &x, &mut plan.sizes());
        assert!(got == want, "chunk plan {k}");
    }
}

#[test]
fn reset_restarts_the_channelizer() {
    let mut p = build(&flow(&CHANNELS));
    let x = signal(3000);
    let _ = run(&mut p, &x[..1001], &mut std::iter::repeat(1001));
    p.reset();
    let got = run(&mut p, &x, &mut std::iter::repeat(3000));
    assert_eq!(got, reference(&x, &CHANNELS));
}
