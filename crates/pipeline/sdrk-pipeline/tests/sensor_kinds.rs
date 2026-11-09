//! The kind `sensor_decode` of M46: a packet to packet node.
//!
//! The kind validates behind a packet input (no `rate_hz` needed) and behind `band_pulse_detect`,
//! the report names its rates and cost, a stream into it is `type_mismatch`, an encoder still
//! cannot read its packets, the flow equals the block on whole buffers and over 20 chunk plans
//! (seed 643), the packet indices stay in the index domain of the pipeline input (also when the
//! input starts at a later index), a chain of `channelize`, `band_pulse_detect` and
//! `sensor_decode` gives the readings of the blocks wired by hand, and nothing allocates after
//! build (a counting global allocator, as in `no_alloc.rs`).
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
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::large_stack_frames
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use sdrk::{C32, Rate};
use sdrk_blocks::{
    BandPulseDetectBlock, BandPulseDetectConfig, Block, ChannelizeBlock, ChannelizeConfig,
    LanesMut, LanesRef, Packet, SensorDecodeBlock, SensorDecodeConfig,
};
use sdrk_device::{BlockMeta, SampleIndex};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Pipeline, Registry, Report};
use sdrk_proto::PacketFlags;
use sdrk_proto::pulses::{self, Header, Modulation, Pair};
use sdrk_testkit::golden::{self, Array};
use sdrk_testkit::ism::{Coding, Placed, Scene, Transmission};
use sdrk_testkit::{ChunkPlan, VecStorage};
use serde_json::{Value, json};

/// The seed of the chunk plans (test 6 of M46).
const SEED_CHUNKS: u64 = 643;

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

/// `trains` (packet input, no rate) -> `dec` (`sensor_decode`) -> `readings`.
fn packet_flow() -> Value {
    json!({
        "sdrk": 1, "name": "sensors_in",
        "inputs": { "trains": { "type": "packets" } },
        "nodes": { "dec": { "kind": "sensor_decode", "params": {
            "emit_unknown": true, "center_hz": 433_920_000 } } },
        "edges": [
            { "from": "trains", "to": "dec" },
            { "from": "dec", "to": "readings" }
        ],
        "outputs": { "readings": { "type": "packets", "queue": 4096 } }
    })
}

/// `iq` (c32 at `rate`) -> `chan` (`channelize`) -> `det` (`band_pulse_detect`) -> `dec`
/// (`sensor_decode`) -> `readings`; `pulses` also takes the trains.
fn chain_flow(rate: u64) -> Value {
    json!({
        "sdrk": 1, "name": "sensor_chain",
        "inputs": { "iq": { "type": "c32", "rate_hz": rate } },
        "nodes": {
            "chan": { "kind": "channelize", "params": { "channels": 16 } },
            "det": { "kind": "band_pulse_detect", "params": { "center_hz": 433_920_000.0 } },
            "dec": { "kind": "sensor_decode", "params": {
                "center_hz": 433_920_000, "emit_unknown": true } }
        },
        "edges": [
            { "from": "iq", "to": "chan" },
            { "from": "chan", "to": "det" },
            { "from": "det", "to": "dec" },
            { "from": "det", "to": "pulses" },
            { "from": "dec", "to": "readings" }
        ],
        "outputs": {
            "pulses": { "type": "packets", "queue": 4096 },
            "readings": { "type": "packets", "queue": 4096 }
        }
    })
}

fn golden_trains(name: &str) -> Vec<Vec<Pair>> {
    let c = golden::load("sensors", name).unwrap();
    let Array::I32(counts) = c.array("pair_counts").unwrap() else {
        panic!("pair_counts");
    };
    let Array::I32(flat) = c.array("pairs").unwrap() else {
        panic!("pairs");
    };
    let mut at = 0;
    let mut out = Vec::new();
    for &n in counts {
        let n = n as usize;
        out.push(
            flat[at..at + 2 * n]
                .chunks_exact(2)
                .map(|w| Pair {
                    pulse: w[0] as u32,
                    gap: w[1] as u32,
                })
                .collect(),
        );
        at += 2 * n;
    }
    out
}

/// Some trains of every decoder, 3 s apart, with copies on other channels and a train of random
/// widths now and then (so that merging, unknown trains and trains of several packets occur).
fn train_packets() -> Vec<Packet> {
    let mut out = Vec::new();
    let mut index = 0u64;
    for name in [
        "wh2",
        "nexus",
        "oregon_v21",
        "wh1080",
        "lacrosse_tx29",
        "bresser_3ch",
    ] {
        let m = if name == "lacrosse_tx29" {
            Modulation::Fsk
        } else {
            Modulation::Ook
        };
        for (k, pairs) in golden_trains(name).iter().take(6).enumerate() {
            for (dt, ch) in [(0u64, 2i16), (20_000, 2), (60_000, -3)] {
                let h = Header {
                    modulation: m,
                    channel: ch,
                    freq_offset_hz: 40_000,
                    level_db10: -300,
                    noise_db10: -700,
                    rate_hz: 250_000,
                };
                out.extend(
                    pulses::packets(index + dt, PacketFlags::NONE, &h, pairs).map(Result::unwrap),
                );
            }
            if k % 2 == 0 {
                let h = Header {
                    modulation: m,
                    channel: 1,
                    freq_offset_hz: 0,
                    level_db10: -300,
                    noise_db10: -700,
                    rate_hz: 250_000,
                };
                let junk: Vec<Pair> = (0..70)
                    .map(|i| Pair {
                        pulse: 20 + i * 3,
                        gap: 40 + i,
                    })
                    .collect();
                out.extend(
                    pulses::packets(index + 100_000, PacketFlags::NONE, &h, &junk)
                        .map(Result::unwrap),
                );
            }
            index += 750_000;
        }
    }
    out
}

/// The readings of the block alone, whole buffer.
fn block_output(packets: &[Packet]) -> Vec<Packet> {
    let cfg = SensorDecodeConfig {
        center_hz: 433_920_000,
        emit_unknown: true,
        ..SensorDecodeConfig::default()
    };
    let mut b = SensorDecodeBlock::new(&cfg, &mut VecStorage).unwrap();
    let mut out = vec![Packet::default(); 8 * packets.len() + 8];
    let p = b.process(packets, &mut out).unwrap();
    assert_eq!(p.consumed, packets.len());
    out.truncate(p.produced);
    out
}

/// Pushes the packets in pieces of `sizes` and reads the readings after each push.
fn run_packets(
    p: &mut Pipeline,
    packets: &[Packet],
    sizes: &mut dyn Iterator<Item = usize>,
) -> Vec<Packet> {
    let mut got = Vec::new();
    let mut buf = vec![Packet::default(); 4096];
    let mut at = 0;
    while at < packets.len() {
        let n = sizes.next().unwrap().min(packets.len() - at);
        p.push_packets("trains", &packets[at..at + n]).unwrap();
        at += n;
        let k = p.read_packets("readings", &mut buf).unwrap();
        got.extend_from_slice(&buf[..k]);
    }
    got
}

#[test]
fn the_kind_is_registered() {
    let registry = Registry::standard();
    let dec = registry.factory("sensor_decode").unwrap();
    assert_eq!(dec.kernels().len(), 0);
    assert_eq!(registry.descriptions().count(), 71);
}

#[test]
fn a_packet_input_into_the_decoder_needs_no_rate() {
    let report = validate(&packet_flow());
    assert!(report.is_ok(), "{}", report.to_json());
    let dec = report.nodes.iter().find(|n| n.name == "dec").unwrap();
    assert_eq!(dec.kind, "sensor_decode");
    // At most 4 readings per train: the nominal packet rate is four times the input's.
    assert_eq!(
        dec.rate_out,
        Rate::hz(4 * dec.rate_in.num() / dec.rate_in.den()).unwrap()
    );
    // A rate on the packet input is accepted too (it sizes the buffers).
    let mut spec = packet_flow();
    spec["inputs"]["trains"]["rate_hz"] = json!(1000);
    let report = validate(&spec);
    assert!(report.is_ok(), "{}", report.to_json());
    let dec = report.nodes.iter().find(|n| n.name == "dec").unwrap();
    assert_eq!(dec.rate_in, Rate::hz(1000).unwrap());
    assert_eq!(dec.rate_out, Rate::hz(4000).unwrap());
    // The cost is 50 per pulse train.
    assert!(
        (report.cost_per_input_sample - 50.0 * 1000.0 / 1000.0).abs() < 1e-9,
        "{}",
        report.cost_per_input_sample
    );
}

#[test]
fn the_decoder_behind_the_detector_validates_and_reports_its_rates() {
    let report = validate(&chain_flow(2_000_000));
    assert!(report.is_ok(), "{}", report.to_json());
    let det = report.nodes.iter().find(|n| n.name == "det").unwrap();
    let dec = report.nodes.iter().find(|n| n.name == "dec").unwrap();
    assert_eq!(dec.rate_in, det.rate_out);
    assert_eq!(dec.rate_out, Rate::hz(4 * 250_000).unwrap());
    assert_eq!(dec.clock_domain, "iq");
    // Channelize 8 + 4 5/2 per input sample, the detector 12 M per step of 8 samples, the decoder
    // 50 per train at the nominal rate of one train per step.
    let want = 18.0 + 12.0 * 16.0 / 8.0 + 50.0 / 8.0;
    assert!(
        (report.cost_per_input_sample - want).abs() < 1e-9,
        "{}",
        report.cost_per_input_sample
    );
}

#[test]
fn a_stream_into_the_decoder_is_a_type_mismatch() {
    let spec = json!({
        "sdrk": 1, "name": "bad",
        "inputs": { "iq": { "type": "c32", "rate_hz": 250_000 } },
        "nodes": { "dec": { "kind": "sensor_decode" } },
        "edges": [ { "from": "iq", "to": "dec" }, { "from": "dec", "to": "readings" } ],
        "outputs": { "readings": { "type": "packets" } }
    });
    let errors = validate(&spec).errors;
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].code, DiagCode::TypeMismatch);
    assert_eq!(errors[0].path, "edges[0].to");
    // Its readings go to packet outputs only.
    let mut spec = packet_flow();
    spec["outputs"]["readings"] = json!({ "type": "stream" });
    let errors = validate(&spec).errors;
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].code, DiagCode::TypeMismatch);
}

#[test]
fn an_encoder_still_cannot_read_the_packets_of_a_decoder() {
    for source in ["dec", "det"] {
        let mut spec = chain_flow(2_000_000);
        spec["nodes"]["enc"] = json!({ "kind": "frame_encode" });
        spec["edges"] = json!([
            { "from": "iq", "to": "chan" },
            { "from": "chan", "to": "det" },
            { "from": "det", "to": "dec" },
            { "from": source, "to": "enc" },
            { "from": "enc", "to": "tx" },
            { "from": "dec", "to": "readings" }
        ]);
        spec["outputs"] = json!({
            "tx": { "type": "stream" },
            "readings": { "type": "packets" }
        });
        let report = validate(&spec);
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.code == DiagCode::TypeMismatch && e.path == "edges[3].to"),
            "{source}: {}",
            report.to_json()
        );
    }
}

#[test]
fn a_bad_decoder_mask_is_reported_at_its_parameter() {
    for (key, value) in [
        ("decoders", json!(0)),
        ("decoders", json!(64)),
        ("merge_s", json!(-1.0)),
    ] {
        let mut spec = packet_flow();
        spec["nodes"]["dec"]["params"][key] = value;
        let errors = validate(&spec).errors;
        assert_eq!(errors.len(), 1, "{key}: {errors:?}");
        assert_eq!(
            errors[0].path,
            format!("nodes.dec.params.{key}"),
            "{errors:?}"
        );
    }
}

#[test]
fn the_flow_gives_the_packets_of_the_block_on_whole_buffers_and_over_20_plans() {
    let trains = train_packets();
    let want = block_output(&trains);
    assert!(want.len() > 30, "{}", want.len());
    assert!(trains.iter().any(|p| p.flags.contains(pulses::CONTINUED)));
    for size in [usize::MAX, 1000, 7] {
        let mut p = build(&packet_flow());
        let got = run_packets(&mut p, &trains, &mut std::iter::repeat(size));
        assert!(got == want, "pieces of {size}");
    }
    for k in 0..20u64 {
        let plan = ChunkPlan::new(SEED_CHUNKS, 1 + 5 * k as usize);
        let mut p = build(&packet_flow());
        let got = run_packets(&mut p, &trains, &mut plan.sizes());
        assert!(got == want, "plan {k}");
        // The indices are those of the trains: nothing is added.
        assert!(got.iter().all(|r| r.index % 750_000 < 100_001), "plan {k}");
    }
}

#[test]
fn a_reset_forgets_the_copies() {
    let trains = train_packets();
    let mut p = build(&packet_flow());
    let first = run_packets(&mut p, &trains, &mut std::iter::repeat(64));
    p.reset();
    let again = run_packets(&mut p, &trains, &mut std::iter::repeat(64));
    assert_eq!(first, again);
}

const NEXUS_BITS: [bool; 36] = {
    // id 0x5c, battery 1, 0, channel 2, 27.1 C, 1111, 56 %: 5c910ff38.
    let nibbles = [0x5u8, 0xc, 0x9, 0x1, 0x0, 0xf, 0xf, 0x3, 0x8];
    let mut bits = [false; 36];
    let mut i = 0;
    while i < 36 {
        bits[i] = (nibbles[i / 4] >> (3 - i % 4)) & 1 == 1;
        i += 1;
    }
    bits
};

/// Four Nexus transmissions (three copies each, 4 ms apart) and an FSK burst at 2 MS/s.
fn scene_samples() -> Vec<C32> {
    let ppm = Coding::Ppm {
        pulse_us: 500.0,
        short_gap_us: 1000.0,
        long_gap_us: 2000.0,
    };
    let tx = |freq: f64, level: f64| Transmission {
        coding: ppm,
        bits: &NEXUS_BITS,
        repeats: 3,
        repeat_gap_us: 4000.0,
        freq_offset_hz: freq,
        level_db: level,
        timing_error: 0.0,
    };
    let placed = [
        Placed {
            start_s: 0.010,
            transmission: tx(-310e3, -25.0),
        },
        Placed {
            start_s: 0.060,
            transmission: tx(190e3, -20.0),
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
    let mut x = vec![C32::default(); 800_000];
    scene.renderer().unwrap().render(&mut x);
    x
}

/// The trains of the detector wired by hand.
fn trains_of(x: &[C32]) -> Vec<Packet> {
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

/// Pushes `x` in pieces of `size` from the index `start` and reads both outputs after each push.
fn run_chain(p: &mut Pipeline, x: &[C32], size: usize, start: u64) -> (Vec<Packet>, Vec<Packet>) {
    let (mut trains, mut readings) = (Vec::new(), Vec::new());
    let mut buf = vec![Packet::default(); 4096];
    let mut index = start;
    for piece in x.chunks(size) {
        let meta = BlockMeta {
            start: SampleIndex(index),
            ..BlockMeta::default()
        };
        p.push_with_meta("iq", InputData::C32(piece), &meta)
            .unwrap();
        index += piece.len() as u64;
        let k = p.read_packets("pulses", &mut buf).unwrap();
        trains.extend_from_slice(&buf[..k]);
        let k = p.read_packets("readings", &mut buf).unwrap();
        readings.extend_from_slice(&buf[..k]);
    }
    (trains, readings)
}

#[test]
fn the_chain_gives_the_readings_of_the_blocks_wired_by_hand() {
    let x = scene_samples();
    let want_trains = trains_of(&x);
    let want = block_output(&want_trains);
    assert!(!want.is_empty(), "the Nexus transmissions are decoded");
    let text = String::from_utf8(want[0].data().to_vec()).unwrap();
    assert!(text.contains("\"model\":\"nexus\""), "{text}");
    assert!(text.contains("\"temperature_C\":27.1"), "{text}");
    assert!(text.contains("\"humidity\":56"), "{text}");
    for size in [4000, 777] {
        let (trains, readings) = run_chain(&mut build(&chain_flow(2_000_000)), &x, size, 0);
        assert!(trains == want_trains, "trains, pieces of {size}");
        assert!(readings == want, "readings, pieces of {size}");
    }
}

/// The readings carry the index of their train in the index domain of the pipeline input, also
/// when the input starts later than 0 (the nodes then start at that index: the packets of a
/// packet to packet node must not be moved a second time).
#[test]
fn the_indices_stay_in_the_domain_of_the_pipeline_input() {
    let x = scene_samples();
    let start = 1_600_000u64;
    let (trains, readings) = run_chain(&mut build(&chain_flow(2_000_000)), &x, 4000, start);
    let want = block_output(&trains);
    assert_ne!(readings.len(), 0);
    assert!(
        readings == want,
        "the readings of the trains the flow reported"
    );
    for r in &readings {
        let train = trains
            .iter()
            .find(|t| t.index == r.index)
            .expect("a train with this index");
        // The indices of the trains are time steps of the channelizer (one per 8 input samples).
        assert!(train.index >= start / 8, "{} < {}", train.index, start / 8);
        assert!(train.index < (start + x.len() as u64) / 8);
    }
    let (_, zero) = run_chain(&mut build(&chain_flow(2_000_000)), &x, 4000, 0);
    assert_eq!(zero.len(), readings.len());
    for (a, b) in zero.iter().zip(&readings) {
        // The packet indices are in the domain of the detector's input (steps).
        assert_eq!(a.index + start / 8, b.index);
        // The time field is derived from the index: it moves by the same amount.
        let ta = String::from_utf8(a.data().to_vec()).unwrap();
        let tb = String::from_utf8(b.data().to_vec()).unwrap();
        assert_ne!(ta, tb);
    }
}

#[test]
fn nothing_allocates_after_build() {
    let x = scene_samples();
    let mut p = build(&chain_flow(2_000_000));
    let mut buf = vec![Packet::default(); 4096];
    // Warm up once outside the measurement.
    p.push("iq", InputData::C32(&x[..1000])).unwrap();
    p.read_packets("pulses", &mut buf).unwrap();
    p.read_packets("readings", &mut buf).unwrap();
    let (produced, allocations) = count_allocations(|| {
        let mut produced = 0usize;
        for piece in x[1000..].chunks(1000 + 37) {
            p.push("iq", InputData::C32(piece)).unwrap();
            p.read_packets("pulses", &mut buf).unwrap();
            produced += p.read_packets("readings", &mut buf).unwrap();
        }
        produced
    });
    assert!(produced > 0, "the pipeline did work");
    assert_eq!(allocations, 0, "allocations after build");
    // The same for the packet flow.
    let trains = train_packets();
    let mut p = build(&packet_flow());
    p.push_packets("trains", &trains[..3]).unwrap();
    p.read_packets("readings", &mut buf).unwrap();
    let (produced, allocations) = count_allocations(|| {
        let mut produced = 0usize;
        for piece in trains[3..].chunks(40) {
            p.push_packets("trains", piece).unwrap();
            produced += p.read_packets("readings", &mut buf).unwrap();
        }
        produced
    });
    assert!(produced > 0);
    assert_eq!(allocations, 0, "allocations after build");
}

#[test]
fn the_counter_notices_allocations() {
    // Guards the test above against a counter that never counts.
    let (_, n) = count_allocations(|| std::hint::black_box(vec![1u8; 100]));
    assert!(n >= 1);
}
