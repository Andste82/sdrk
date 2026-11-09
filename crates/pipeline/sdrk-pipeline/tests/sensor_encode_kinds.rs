//! The kinds `sensor_encode` and `pulse_key` of the M46 extension: the chain `reading` (packet
//! input) -> `sensor_encode` -> `pulse_key` (port b, the clock at port a) -> `out`.
//!
//! The packets of a node reach port b of a clocked node (no change of the scheduler was needed):
//! the chain validates without a rate on the packet input, equals the blocks wired by hand in
//! push mode over 20 chunk plans (seed 774), works in pull mode with packets pushed between the
//! pulls, reports the readings of both nodes, resets, rejects a stream at either packet port
//! (`type_mismatch`), keeps the rule that an encoder cannot read the packets of a decoder, and
//! allocates nothing after build (a counting global allocator, as in `sensor_kinds.rs`).
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
    clippy::large_stack_frames
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use sdrk::Rate;
use sdrk_blocks::{
    Block, Packet, PulseKeyBlock, PulseKeyConfig, SensorEncodeBlock, SensorEncodeConfig,
};
use sdrk_pipeline::{
    BuildOptions, DiagCode, InputData, Pipeline, PullError, PullSource, Registry, Report,
};
use sdrk_proto::PacketFlags;
use sdrk_testkit::{ChunkPlan, VecStorage};
use serde_json::{Value, json};

/// The seed of the chunk plans.
const SEED_CHUNKS: u64 = 774;

const NEXUS: &str =
    r#"{"model":"nexus","id":92,"channel":2,"battery_ok":true,"temperature_C":-3.5,"humidity":71}"#;
const BRESSER: &str = r#"{"model":"Bresser-3CH","id":66,"channel":1,"battery_ok":true,"temperature_C":22.56,"humidity":66}"#;

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

fn validate(spec: &Value) -> Report {
    Registry::standard().validate_json(&spec.to_string(), &BuildOptions::default())
}

fn build(spec: &Value) -> Pipeline {
    Registry::standard()
        .build_json(&spec.to_string(), &BuildOptions::default())
        .unwrap()
}

/// `clock` (f32 at 48 kHz) and `reading` (packets, no rate) -> `enc` -> `key` (port b) -> `out`.
fn flow() -> Value {
    json!({
        "sdrk": 1, "name": "sensor_tx", "settings": { "precision": "f32", "max_chunk": 512 },
        "inputs": {
            "clock": { "type": "f32", "rate_hz": 48_000 },
            "reading": { "type": "packets" }
        },
        "nodes": {
            "enc": { "kind": "sensor_encode", "params": {} },
            "key": { "kind": "pulse_key", "params": {} }
        },
        "edges": [
            { "from": "clock", "to": "key" },
            { "from": "reading", "to": "enc" },
            { "from": "enc", "to": "key", "port": "b" },
            { "from": "key", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    })
}

fn reading(index: u64, text: &str) -> Packet {
    Packet::new(index, PacketFlags::NONE, text.as_bytes()).unwrap()
}

/// The envelope of the blocks wired by hand: all packets at the first call, `n` samples.
fn blocks_envelope(text: &str, n: usize) -> Vec<f32> {
    let mut enc = SensorEncodeBlock::new(&SensorEncodeConfig::default(), &mut VecStorage).unwrap();
    let mut trains = vec![Packet::default(); 64];
    let p = enc.process(&[reading(0, text)], &mut trains).unwrap();
    let rate = Rate::hz(48_000).unwrap();
    let mut key = PulseKeyBlock::new(&PulseKeyConfig::default(), rate, &mut VecStorage).unwrap();
    let clock = vec![0.0f32; n];
    let mut out = vec![0.0f32; n];
    key.process((&clock, &trains[..p.produced]), &mut out)
        .unwrap();
    out
}

/// Pushes the reading and then the clock in pieces of `sizes`, and reads the envelope.
fn push_envelope(
    p: &mut Pipeline,
    text: &str,
    n: usize,
    sizes: &mut dyn Iterator<Item = usize>,
) -> Vec<f32> {
    p.push_packets("reading", &[reading(0, text)]).unwrap();
    let clock = vec![0.0f32; 4096];
    let mut got = Vec::new();
    let mut buf = vec![0.0f32; 4096];
    while got.len() < n {
        let k = sizes.next().unwrap().min(n - got.len()).min(clock.len());
        p.push("clock", InputData::F32(&clock[..k])).unwrap();
        loop {
            let m = p.read_stream::<f32>("out", &mut buf).unwrap();
            if m == 0 {
                break;
            }
            got.extend_from_slice(&buf[..m]);
        }
    }
    got
}

#[test]
fn the_chain_validates_without_a_rate_on_the_packet_input() {
    let report = validate(&flow());
    assert!(report.is_ok(), "{}", report.to_json());
    let enc = report.nodes.iter().find(|n| n.name == "enc").unwrap();
    let key = report.nodes.iter().find(|n| n.name == "key").unwrap();
    assert_eq!(enc.kind, "sensor_encode");
    assert_eq!(key.kind, "pulse_key");
    // The envelope has the rate of the clock; the packet stream has a nominal rate (a bound for
    // the buffers: 40 packets per reading).
    assert_eq!(key.rate_out, Rate::hz(48_000).unwrap());
    assert!(enc.rate_out.as_f64() >= 40.0 * enc.rate_in.as_f64());
    // The memory holds the train of the encoder (9 600 bytes) and the queue of 4 trains of the
    // key (38 400 bytes) and the streams.
    assert!(
        report.memory_bytes >= 9600 + 38_400,
        "{}",
        report.memory_bytes
    );
}

#[test]
fn the_chain_equals_the_blocks_in_push_mode_over_20_plans() {
    for text in [NEXUS, BRESSER] {
        let n = 60_000;
        let want = blocks_envelope(text, n);
        assert!(want.iter().any(|&y| y > 0.0) && want.iter().any(|&y| y < 0.0));
        let mut p = build(&flow());
        let got = push_envelope(&mut p, text, n, &mut std::iter::repeat(512));
        assert!(got == want, "512 {text}");
        for k in 0..20u64 {
            let plan = ChunkPlan::new(SEED_CHUNKS, 1 + 25 * k as usize);
            let mut p = build(&flow());
            let got = push_envelope(&mut p, text, n, &mut plan.sizes());
            assert!(got == want, "plan {k} {text}");
        }
    }
}

/// A clock of zeros for pull mode.
struct Zeros;

impl PullSource for Zeros {
    fn provide(&mut self, input: &str, max: usize) -> Result<InputData<'_>, PullError> {
        static DATA: [f32; 4096] = [0.0; 4096];
        match input {
            "clock" => Ok(InputData::F32(&DATA[..max.min(4096)])),
            _ => Ok(InputData::Packets(&[])),
        }
    }
}

#[test]
fn the_chain_works_in_pull_mode_with_packets_pushed_between_the_pulls() {
    let mut p = build(&flow());
    let mut buf = vec![0.0f32; 480];
    // Idle: the carrier is off.
    let (n, _) = p.pull_samples::<f32>("out", &mut buf, &mut Zeros).unwrap();
    assert_eq!(n, 480);
    assert!(buf.iter().all(|&y| y == -1.0));
    // A reading pushed between two pulls is keyed from the next pull on.
    p.push_packets("reading", &[reading(0, NEXUS)]).unwrap();
    let mut got = Vec::new();
    while got.len() < 60_000 {
        let (n, _) = p.pull_samples::<f32>("out", &mut buf, &mut Zeros).unwrap();
        assert_eq!(n, 480);
        got.extend_from_slice(&buf);
    }
    let on = got.iter().filter(|&&y| y > 0.0).count();
    assert!(on > 100, "{on}");
    assert!(got.iter().all(|&y| y.abs() == 1.0));
    // The same envelope as the blocks, the train starting with the first pull after the push.
    let want = blocks_envelope(NEXUS, 60_000);
    // The pull path may have a sample or two queued before the packet reaches the key, so the
    // train starts that late; the shape is the same.
    let first = |x: &[f32]| x.iter().position(|&y| y > 0.0).unwrap();
    let late = first(&got) - first(&want);
    assert!(late <= 4, "{late}");
    assert_eq!(got[late..], want[..got.len() - late]);
    let stats = p.stats();
    let key = stats.iter().find(|s| s.name == "key").unwrap();
    let enc = stats.iter().find(|s| s.name == "enc").unwrap();
    assert_eq!(key.readings[2], ("trains_sent", 1.0));
    assert_eq!(enc.readings[0], ("encoded", 1.0));
    // A second reading follows the first one.
    p.push_packets("reading", &[reading(1, NEXUS), reading(2, "not json")])
        .unwrap();
    for _ in 0..125 {
        p.pull_samples::<f32>("out", &mut buf, &mut Zeros).unwrap();
    }
    let stats = p.stats();
    let enc = stats.iter().find(|s| s.name == "enc").unwrap();
    let key = stats.iter().find(|s| s.name == "key").unwrap();
    assert_eq!(enc.readings[0], ("encoded", 2.0));
    assert_eq!(enc.readings[1], ("invalid", 1.0));
    assert_eq!(key.readings[2], ("trains_sent", 2.0));
}

#[test]
fn a_reset_forgets_the_trains() {
    let mut p = build(&flow());
    let first = push_envelope(&mut p, NEXUS, 20_000, &mut std::iter::repeat(300));
    assert!(first.iter().any(|&y| y > 0.0));
    p.reset();
    let stats = p.stats();
    let key = stats.iter().find(|s| s.name == "key").unwrap();
    assert_eq!(key.readings[2], ("trains_sent", 0.0));
    let second = push_envelope(&mut p, NEXUS, 20_000, &mut std::iter::repeat(300));
    assert_eq!(first, second);
}

#[test]
fn a_stream_at_either_packet_port_is_a_type_mismatch() {
    // The clock into port b of pulse_key.
    let mut spec = flow();
    spec["edges"] = json!([
        { "from": "clock", "to": "key" },
        { "from": "reading", "to": "enc" },
        { "from": "clock", "to": "key", "port": "b" },
        { "from": "enc", "to": "trains" },
        { "from": "key", "to": "out" }
    ]);
    spec["outputs"]["trains"] = json!({ "type": "packets" });
    let errors = validate(&spec).errors;
    assert!(
        errors.iter().any(|e| e.code == DiagCode::TypeMismatch),
        "{errors:?}"
    );
    // A stream into sensor_encode.
    let mut spec = flow();
    spec["edges"] = json!([
        { "from": "clock", "to": "key" },
        { "from": "clock", "to": "enc" },
        { "from": "enc", "to": "key", "port": "b" },
        { "from": "key", "to": "out" }
    ]);
    let errors = validate(&spec).errors;
    assert!(
        errors
            .iter()
            .any(|e| e.code == DiagCode::TypeMismatch && e.path == "edges[1].to"),
        "{errors:?}"
    );
    // Packets at the clock port a of pulse_key.
    let mut spec = flow();
    spec["edges"] = json!([
        { "from": "reading", "to": "key" },
        { "from": "reading", "to": "enc" },
        { "from": "enc", "to": "key", "port": "b" },
        { "from": "key", "to": "out" }
    ]);
    let errors = validate(&spec).errors;
    assert!(
        errors.iter().any(|e| e.code == DiagCode::TypeMismatch),
        "{errors:?}"
    );
}

#[test]
fn the_parameters_are_checked_at_their_name() {
    let mut spec = flow();
    spec["nodes"]["enc"]["params"] = json!({ "repeats": 65 });
    let errors = validate(&spec).errors;
    assert!(
        errors.iter().any(|e| e.path.contains("repeats")),
        "{errors:?}"
    );
    let mut spec = flow();
    spec["nodes"]["key"]["params"] = json!({ "guard_us": -1 });
    let errors = validate(&spec).errors;
    assert!(
        errors.iter().any(|e| e.path.contains("guard_us")),
        "{errors:?}"
    );
    // A parameter that changes the result is accepted.
    let mut spec = flow();
    spec["nodes"]["enc"]["params"] = json!({ "repeats": 2 });
    spec["nodes"]["key"]["params"] = json!({ "guard_us": 0 });
    assert!(validate(&spec).is_ok());
}

#[test]
fn nothing_allocates_after_build() {
    let mut p = build(&flow());
    let clock = vec![0.0f32; 480];
    let mut buf = vec![0.0f32; 4096];
    p.push_packets("reading", &[reading(0, NEXUS)]).unwrap();
    p.push("clock", InputData::F32(&clock)).unwrap();
    p.read_stream::<f32>("out", &mut buf).unwrap();
    let (produced, allocations) = count_allocations(|| {
        let mut produced = 0usize;
        for k in 0..200 {
            if k % 50 == 10 {
                p.push_packets("reading", &[reading(k, BRESSER)]).unwrap();
            }
            p.push("clock", InputData::F32(&clock)).unwrap();
            produced += p.read_stream::<f32>("out", &mut buf).unwrap();
        }
        produced
    });
    assert!(produced > 0, "the pipeline did work");
    assert_eq!(allocations, 0, "allocations after build");
    // The same in pull mode.
    let mut p = build(&flow());
    let mut out = vec![0.0f32; 480];
    p.pull_samples::<f32>("out", &mut out, &mut Zeros).unwrap();
    let ((), allocations) = count_allocations(|| {
        for k in 0..100u64 {
            if k % 40 == 3 {
                p.push_packets("reading", &[reading(k, NEXUS)]).unwrap();
            }
            p.pull_samples::<f32>("out", &mut out, &mut Zeros).unwrap();
        }
    });
    assert_eq!(allocations, 0, "allocations after build (pull)");
}

#[test]
fn the_counter_notices_allocations() {
    // Guards the test above against a counter that never counts.
    let (_, n) = count_allocations(|| std::hint::black_box(vec![1u8; 100]));
    assert!(n >= 1);
}
