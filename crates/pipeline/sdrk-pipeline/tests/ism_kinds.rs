//! The kind `band_pulse_detect` of M45.1 (spec test 1).
//!
//! A `channelize` node into `band_pulse_detect` validates and builds, the report names the step
//! rate, the packet rate and the cost, a stream into `band_pulse_detect` is `type_mismatch`, a
//! step rate below 100 kHz is `rate_unsupported` at the node, the packets of the flow equal
//! those of a `ChannelizeBlock` and a `BandPulseDetectBlock` wired by hand, the runtime parameter
//! `min_snr_db` reaches the block, and nothing allocates after build (a counting global
//! allocator, as in `no_alloc.rs`). The scenes are fixed (no seed of the spec: seeds 629 to 635
//! belong to tests 3 to 8).
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

use sdrk::{C32, Rate};
use sdrk_blocks::{
    BandPulseDetectBlock, BandPulseDetectConfig, Block, ChannelizeBlock, ChannelizeConfig,
    LanesMut, LanesRef, Packet,
};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Pipeline, Registry, Report};
use sdrk_proto::pulses::{self, Modulation};
use sdrk_testkit::VecStorage;
use sdrk_testkit::ism::{Coding, Placed, Scene, Transmission};
use serde_json::{Value, json};

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

/// `iq` (c32 at `rate`) -> `chan` (`channelize`, `channels`) -> `det` (`band_pulse_detect`) ->
/// `pulses` (packets).
fn flow(rate: u64, channels: usize) -> Value {
    json!({
        "sdrk": 1, "name": "band_pulses",
        "inputs": { "iq": { "type": "c32", "rate_hz": rate } },
        "nodes": {
            "chan": { "kind": "channelize", "params": { "channels": channels } },
            "det": { "kind": "band_pulse_detect", "params": { "center_hz": 433_920_000.0 } }
        },
        "edges": [
            { "from": "iq", "to": "chan" },
            { "from": "chan", "to": "det" },
            { "from": "det", "to": "pulses" }
        ],
        "outputs": { "pulses": { "type": "packets", "queue": 4096 } }
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

const BITS: [bool; 16] = [
    true, false, false, true, true, true, false, true, false, false, true, false, true, true,
    false, false,
];

/// Two PWM transmissions and one FSK transmission in the 433 MHz band at 2 MHz, 50 ms.
fn scene_samples() -> Vec<C32> {
    let pwm = Coding::Pwm {
        short_us: 200.0,
        long_us: 400.0,
        gap_us: 200.0,
    };
    let tx = |coding, freq: f64, level: f64| Transmission {
        coding,
        bits: &BITS,
        repeats: 2,
        repeat_gap_us: 2000.0,
        freq_offset_hz: freq,
        level_db: level,
        timing_error: 0.0,
    };
    let placed = [
        Placed {
            start_s: 0.006,
            transmission: tx(pwm, -310e3, -25.0),
        },
        Placed {
            start_s: 0.012,
            transmission: tx(pwm, 190e3, -20.0),
        },
        Placed {
            start_s: 0.030,
            transmission: tx(
                Coding::Fsk {
                    bit_us: 100.0,
                    deviation_hz: 30e3,
                },
                560e3,
                -22.0,
            ),
        },
    ];
    let scene = Scene {
        fs: 2e6,
        center_hz: 433.92e6,
        clock_error_ppm: 0.0,
        noise_db: -30.0,
        transmissions: &placed,
        interferers: &[],
        seed: 99,
    };
    let mut x = vec![C32::default(); 100_000];
    scene.renderer().unwrap().render(&mut x);
    x
}

/// The packets of `x` from the blocks wired by hand.
fn reference(x: &[C32]) -> Vec<Packet> {
    let mut chan = ChannelizeBlock::new(&ChannelizeConfig::default(), &mut VecStorage).unwrap();
    let mut lanes = vec![C32::default(); x.len() / 8 * 16];
    let p = chan
        .process(x, LanesMut::new(&mut lanes, 16).unwrap())
        .unwrap();
    assert_eq!(p.produced, x.len() / 8);
    let cfg = BandPulseDetectConfig {
        center_hz: 433.92e6,
        ..BandPulseDetectConfig::default()
    };
    let mut det =
        BandPulseDetectBlock::new(&cfg, Rate::hz(250_000).unwrap(), 16, &mut VecStorage).unwrap();
    let mut out = vec![Packet::default(); 4096];
    let p = det
        .process(LanesRef::new(&lanes, 16).unwrap(), &mut out)
        .unwrap();
    assert_eq!(p.consumed, x.len() / 8);
    out.truncate(p.produced);
    out
}

/// Pushes `x` in pieces of `size` and reads the packets after each push.
fn run(p: &mut Pipeline, x: &[C32], size: usize) -> Vec<Packet> {
    let mut got = Vec::new();
    let mut buf = vec![Packet::default(); 4096];
    for piece in x.chunks(size) {
        p.push("iq", InputData::C32(piece)).unwrap();
        let k = p.read_packets("pulses", &mut buf).unwrap();
        got.extend_from_slice(&buf[..k]);
    }
    got
}

#[test]
fn the_kind_is_registered() {
    let registry = Registry::standard();
    let det = registry.factory("band_pulse_detect").unwrap();
    assert_eq!(det.kernels(), ["mag"]);
    assert_eq!(registry.descriptions().count(), 66);
}

#[test]
fn a_channelizer_into_the_detector_validates_and_reports_its_rates() {
    let report = validate(&flow(2_000_000, 16));
    assert!(report.is_ok(), "{}", report.to_json());
    let det = report.nodes.iter().find(|n| n.name == "det").unwrap();
    assert_eq!(det.kind, "band_pulse_detect");
    assert_eq!(det.lanes, None);
    // The step rate in, the nominal packet rate M / 16 per step out.
    assert_eq!(det.rate_in, Rate::hz(250_000).unwrap());
    assert_eq!(det.rate_out, Rate::hz(250_000).unwrap());
    assert_eq!(det.clock_domain, "iq");
    // Cost: channelize 8 + 4 5/2 per input sample, the detector 12 M per step of 8 samples.
    let want = 18.0 + 12.0 * 16.0 / 8.0;
    assert!(
        (report.cost_per_input_sample - want).abs() < 1e-9,
        "{}",
        report.cost_per_input_sample
    );
    let json: Value = serde_json::from_str(&report.to_json()).unwrap();
    let nodes = json["nodes"].as_array().unwrap();
    let entry = nodes.iter().find(|n| n["name"] == "det").unwrap();
    assert_eq!(entry["rate_in"], 250_000);
    assert!(entry.get("lanes").is_none());
    // 8 channels at 2 MHz: 500 kHz steps, half a packet per step.
    let report = validate(&flow(2_000_000, 8));
    let det = report.nodes.iter().find(|n| n.name == "det").unwrap();
    assert_eq!(det.rate_in, Rate::hz(500_000).unwrap());
    assert_eq!(det.rate_out, Rate::hz(250_000).unwrap());
}

#[test]
fn a_stream_into_the_detector_is_a_type_mismatch() {
    let spec = json!({
        "sdrk": 1, "name": "bad",
        "inputs": { "iq": { "type": "c32", "rate_hz": 250_000 } },
        "nodes": { "det": { "kind": "band_pulse_detect" } },
        "edges": [ { "from": "iq", "to": "det" }, { "from": "det", "to": "pulses" } ],
        "outputs": { "pulses": { "type": "packets" } }
    });
    let errors = validate(&spec).errors;
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].code, DiagCode::TypeMismatch);
    assert_eq!(errors[0].path, "edges[0].to");
    // Its packets go to packet outputs only.
    let mut spec = flow(2_000_000, 16);
    spec["outputs"]["pulses"] = json!({ "type": "stream" });
    let errors = validate(&spec).errors;
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].code, DiagCode::TypeMismatch);
}

#[test]
fn a_step_rate_below_100_khz_is_rate_unsupported() {
    // 16 channels from 400 kHz: 50 kHz steps.
    let errors = validate(&flow(400_000, 16)).errors;
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].code, DiagCode::RateUnsupported);
    assert_eq!(errors[0].path, "nodes.det");
    assert!(
        errors[0].message.contains("100 kHz"),
        "{}",
        errors[0].message
    );
    // 800 kHz: 100 kHz steps, the smallest step rate.
    assert!(validate(&flow(800_000, 16)).is_ok());
}

#[test]
fn the_flow_gives_the_packets_of_the_blocks_wired_by_hand() {
    let x = scene_samples();
    let want = reference(&x);
    let trains: Vec<_> = want
        .iter()
        .filter(|p| !p.flags.contains(pulses::CONTINUED))
        .map(|p| pulses::decode(p.data()).unwrap().header())
        .collect();
    // The two PWM transmissions (two copies each, one train per copy: the repeat gap is longer
    // than 10 times the longest pulse) and the two frames of the FSK transmission.
    let ook = trains
        .iter()
        .filter(|h| h.modulation == Modulation::Ook)
        .count();
    let fsk = trains
        .iter()
        .filter(|h| h.modulation == Modulation::Fsk)
        .count();
    assert_eq!((ook, fsk), (2, 2), "{trains:?}");
    for size in [4096, 1000, 777] {
        let got = run(&mut build(&flow(2_000_000, 16)), &x, size);
        assert!(got == want, "pieces of {size}");
    }
}

#[test]
fn the_runtime_parameter_reaches_the_block() {
    let x = scene_samples();
    let mut p = build(&flow(2_000_000, 16));
    p.set_param("det", "min_snr_db", json!(60.0)).unwrap();
    assert_eq!(run(&mut p, &x, 4096), Vec::<Packet>::new());
    assert!(p.set_param("det", "max_gap_ms", json!(5.0)).is_err());
}

#[test]
fn nothing_allocates_after_build() {
    let x = scene_samples();
    let mut p = build(&flow(2_000_000, 16));
    let mut buf = vec![Packet::default(); 4096];
    // Warm up once outside the measurement.
    p.push("iq", InputData::C32(&x[..1000])).unwrap();
    p.read_packets("pulses", &mut buf).unwrap();
    let (produced, allocations) = count_allocations(|| {
        let mut produced = 0usize;
        for (i, piece) in x[1000..].chunks(1000 + 37).enumerate() {
            p.push("iq", InputData::C32(piece)).unwrap();
            produced += p.read_packets("pulses", &mut buf).unwrap();
            if i % 20 == 0 {
                p.set_param("det", "min_snr_db", Value::from(9.0)).unwrap();
            }
        }
        produced
    });
    assert!(produced > 0, "the pipeline did work");
    assert_eq!(allocations, 0, "allocations after build");
}

#[test]
fn the_counter_notices_allocations() {
    // Guards the test above against a counter that never counts.
    let (_, n) = count_allocations(|| std::hint::black_box(vec![1u8; 100]));
    assert!(n >= 1);
}
