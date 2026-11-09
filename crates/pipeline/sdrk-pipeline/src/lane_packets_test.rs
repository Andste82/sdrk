//! A lane input on a packet decoder (M45.1 section 3, the pipeline part of spec test 1).
//!
//! The tests register two small test kinds: `lane_fan` (`C32` -> `Lanes<C32>`, one step of
//! `channels` lanes per two input samples, lane `k` of step `i` is `x[2i]·exp(2πjk/channels) + x[2i+1]`)
//! and `lane_burst` (`Lanes<C32>` -> `Packets`, wrapped in `PacketOutNode`): at most one
//! packet per two steps (timing `Packets { max_per_input: 1/2 }`), emitted at a step whose
//! loudest lane differs from that of the step before and has a power of at least `min_power`
//! (a runtime parameter); the packet index is the step, the payload the loudest lane and the
//! width of the input. The seed 451 is local to these tests.
//!
//! The no-allocation test of the wrapper runs with the real kind `band_pulse_detect` in the
//! integration test `tests/ism_kinds.rs`, which has its own counting global allocator, so the
//! crate forbids `unsafe` in every build (D1022).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::needless_pass_by_value,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use sdrk::{C32, Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_blocks::{
    Block, BlockDescription, BlockTiming, LaneElem, Lanes, LanesMut, LanesRef, Mutability, Packet,
    PacketFlags, Packets, ParamDesc, ParamKind, ParamValue, PortDesc, PortType, Unit,
};
use sdrk_device::{BlockMeta, SampleIndex};
use sdrk_testkit::{ChunkPlan, Pcg32};
use serde_json::{Value, json};

use crate::kinds::{Factory, Figures, Kind};
use crate::model::Precision;
use crate::node::{LanesNode, Node, PacketOutNode};
use crate::registry::{BuildOptions, PortKind, Registry};
use crate::runner::{InputData, Pipeline};
use crate::storage::VecStorage;
use crate::validate::DiagCode;

const SEED: u64 = 451;

/// Steps after a packet before the next one may start (so at most one packet per two steps).
const HOLDOFF: u32 = 2;

/// The default of `min_power`.
const MIN_POWER: f64 = 10.0;

// ------------------------------------------------------------------ test kinds

static FAN: BlockDescription = BlockDescription {
    kind: "lane_fan",
    summary: "Test block: one step of lanes per two input samples.",
    details: "Lane k of step i is x[2i]·exp(2πjk/channels) + x[2i+1].",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "The input.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::Lanes {
            elem: LaneElem::C32,
            width_param: "channels",
        }],
        description: "The lanes.",
    }],
    params: &[ParamDesc {
        name: "channels",
        unit: Unit::None,
        kind: ParamKind::I64 {
            min: 1,
            max: 256,
            default: Some(4),
        },
        typical: "16",
        mutability: Mutability::Build,
        description: "Number of lanes.",
    }],
    required: &[],
};

static BURST: BlockDescription = BlockDescription {
    kind: "lane_burst",
    summary: "Test block: a packet when the loudest lane changes.",
    details: "At most one packet per two steps; the index is the step.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::Lanes {
            elem: LaneElem::C32,
            width_param: "",
        }],
        description: "The lanes.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::Packets],
        description: "The packets.",
    }],
    params: &[ParamDesc {
        name: "min_power",
        unit: Unit::None,
        kind: ParamKind::F64 {
            min: 0.0,
            max: 1e9,
            default: Some(MIN_POWER),
        },
        typical: "10",
        mutability: Mutability::Runtime,
        description: "Smallest power of the loudest lane that emits a packet.",
    }],
    required: &[],
};

fn no_param() -> sdrk_blocks::Error {
    sdrk_blocks::Error::UnknownParameter
}

/// The value of lane `k` of `width` lanes of a step from the input pair `(a, b)`: the lane
/// whose rotation of `a` lines up with `b` is the loudest, so every lane can be.
fn fan_value(a: C32, b: C32, k: usize, width: usize) -> C32 {
    a * C32::from_polar(1.0, core::f32::consts::TAU * k as f32 / width as f32) + b
}

#[derive(Debug)]
struct Fan {
    width: usize,
}

impl Block for Fan {
    type In = C32;
    type Out = Lanes<C32>;

    fn description(&self) -> &'static BlockDescription {
        &FAN
    }

    fn process(
        &mut self,
        input: &[C32],
        mut output: LanesMut<'_, C32>,
    ) -> sdrk_blocks::Result<Progress> {
        let steps = (input.len() / 2).min(output.steps());
        let width = self.width;
        for (i, pair) in input.chunks_exact(2).take(steps).enumerate() {
            let step = output.step_mut(i).ok_or(no_param())?;
            for (k, v) in step.iter_mut().enumerate() {
                *v = fan_value(pair[0], pair[1], k, width);
            }
        }
        Ok(Progress {
            consumed: 2 * steps,
            produced: steps,
        })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: Ratio::new(1, 2).unwrap(),
            kind: RatioKind::Constant,
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {}

    fn set_param(&mut self, _name: &str, _value: ParamValue) -> sdrk_blocks::Result<()> {
        Err(no_param())
    }

    fn param(&self, name: &str) -> sdrk_blocks::Result<ParamValue> {
        match name {
            "channels" => Ok(ParamValue::I64(self.width as i64)),
            _ => Err(no_param()),
        }
    }

    fn min_input(&self) -> usize {
        2
    }
}

/// The detector state, shared by the block and the reference.
#[derive(Debug, Clone, Copy)]
struct BurstState {
    min_power: f32,
    /// The loudest lane of the step before (`usize::MAX` before the first step).
    prev: usize,
    /// Steps since the last packet.
    since: u32,
    /// Steps since construction or reset.
    steps: u64,
}

impl BurstState {
    fn new(min_power: f64) -> Self {
        Self {
            min_power: min_power as f32,
            prev: usize::MAX,
            since: HOLDOFF,
            steps: 0,
        }
    }

    /// One step: the loudest lane if the step emits a packet.
    fn step(&mut self, lanes: &[C32]) -> Option<usize> {
        let mut best = (0, -1.0f32);
        for (k, v) in lanes.iter().enumerate() {
            if v.norm_sqr() > best.1 {
                best = (k, v.norm_sqr());
            }
        }
        let emit = best.0 != self.prev && best.1 >= self.min_power && self.since >= HOLDOFF;
        self.prev = best.0;
        self.since = if emit {
            1
        } else {
            self.since.saturating_add(1)
        };
        self.steps += 1;
        emit.then_some(best.0)
    }
}

#[derive(Debug)]
struct Burst {
    state: BurstState,
    min_power: f64,
}

impl Block for Burst {
    type In = Lanes<C32>;
    type Out = Packets;

    fn description(&self) -> &'static BlockDescription {
        &BURST
    }

    fn process(
        &mut self,
        input: LanesRef<'_, C32>,
        output: &mut [Packet],
    ) -> sdrk_blocks::Result<Progress> {
        let width = input.width();
        let mut produced = 0;
        let mut consumed = 0;
        // A step emits at most one packet: stop when the output is full.
        while consumed < input.steps() && produced < output.len() {
            let index = self.state.steps;
            let lanes = input.step(consumed).ok_or(no_param())?;
            if let Some(k) = self.state.step(lanes) {
                output[produced] =
                    Packet::new(index, PacketFlags::NONE, &[k as u8, width as u8]).unwrap();
                produced += 1;
            }
            consumed += 1;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        let max = Ratio::new(1, u64::from(HOLDOFF)).unwrap();
        BlockTiming {
            ratio: max,
            kind: RatioKind::Packets { max_per_input: max },
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
        self.state = BurstState::new(self.min_power);
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> sdrk_blocks::Result<()> {
        match name {
            "min_power" => {
                self.min_power = value.as_f64()?;
                self.state.min_power = self.min_power as f32;
                Ok(())
            }
            _ => Err(no_param()),
        }
    }

    fn param(&self, name: &str) -> sdrk_blocks::Result<ParamValue> {
        match name {
            "min_power" => Ok(ParamValue::F64(self.min_power)),
            _ => Err(no_param()),
        }
    }

    fn min_input(&self) -> usize {
        1
    }
}

struct FanKind;

impl Kind for FanKind {
    type Cfg = usize;

    const KIND: &'static str = "lane_fan";
    fn kernels() -> &'static [&'static str] {
        &[]
    }
    fn default_cfg() -> usize {
        4
    }
    fn set(cfg: &mut usize, name: &str, value: ParamValue) -> sdrk_blocks::Result<()> {
        match name {
            "channels" => {
                *cfg = usize::try_from(value.as_i64()?).map_err(|_| no_param())?;
                Ok(())
            }
            _ => Err(no_param()),
        }
    }
    fn plan(
        cfg: &usize,
        _input: PortType,
        rate: Rate,
        _p: Precision,
    ) -> sdrk_blocks::Result<Figures> {
        Ok(Figures {
            rate_out: rate.scale(Ratio::new(1, 2)?)?,
            memory_bytes: 0,
            cost: *cfg as f64 / 2.0,
            taps: 0,
            delay: Delay::ZERO,
            min_input: 2,
            output: PortKind::Lanes {
                elem: PortType::C32,
                width: *cfg,
            },
        })
    }
    fn build(
        cfg: &usize,
        _input: PortType,
        _rate: Rate,
        _p: Precision,
        _st: &mut VecStorage,
    ) -> sdrk_blocks::Result<Box<dyn Node>> {
        Ok(Box::new(LanesNode::new(Fan { width: *cfg })))
    }
}

/// The threshold and the width of the input.
#[derive(Debug, Clone, Copy)]
struct BurstCfg {
    min_power: f64,
    width: usize,
}

struct BurstKind;

impl Kind for BurstKind {
    type Cfg = BurstCfg;

    const KIND: &'static str = "lane_burst";
    fn kernels() -> &'static [&'static str] {
        &[]
    }
    fn default_cfg() -> BurstCfg {
        BurstCfg {
            min_power: MIN_POWER,
            width: 0,
        }
    }
    fn set(cfg: &mut BurstCfg, name: &str, value: ParamValue) -> sdrk_blocks::Result<()> {
        match name {
            "min_power" => {
                cfg.min_power = value.as_f64()?;
                Ok(())
            }
            _ => Err(no_param()),
        }
    }
    fn input_width(cfg: &mut BurstCfg, width: usize) -> sdrk_blocks::Result<()> {
        cfg.width = width;
        Ok(())
    }
    fn plan(
        cfg: &BurstCfg,
        input: PortType,
        rate: Rate,
        _p: Precision,
    ) -> sdrk_blocks::Result<Figures> {
        assert!(input.is_lanes(), "the factory passes the lane port type");
        assert!(cfg.width > 0, "the factory passes the width first");
        Ok(Figures {
            // The packet stream gets the nominal rate `rate * max_per_input` (steps / 2).
            rate_out: rate.scale(Ratio::new(1, u64::from(HOLDOFF))?)?,
            memory_bytes: 0,
            // Per input element of the node, that is per step of `width` lanes.
            cost: 3.0 * cfg.width as f64,
            taps: 0,
            delay: Delay::ZERO,
            min_input: 1,
            output: PortKind::Stream(PortType::Packets),
        })
    }
    fn build(
        cfg: &BurstCfg,
        _input: PortType,
        _rate: Rate,
        _p: Precision,
        _st: &mut VecStorage,
    ) -> sdrk_blocks::Result<Box<dyn Node>> {
        Ok(Box::new(PacketOutNode::new(Burst {
            state: BurstState::new(cfg.min_power),
            min_power: cfg.min_power,
        })))
    }
}

fn registry() -> Registry {
    let mut r = Registry::standard();
    r.insert_factory(Factory::<FanKind>::boxed(&FAN));
    r.insert_factory(Factory::<BurstKind>::boxed(&BURST));
    r
}

// ------------------------------------------------------------------ helpers

/// `iq` (c32 at 48 kHz) -> `fan` (`lane_fan`, `channels` lanes) -> `det` (`lane_burst`) ->
/// `pkts` (packets).
fn flow(channels: usize) -> Value {
    json!({
        "sdrk": 1, "name": "lane_packets",
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": {
            "fan": { "kind": "lane_fan", "params": { "channels": channels } },
            "det": { "kind": "lane_burst" }
        },
        "edges": [
            { "from": "iq", "to": "fan" },
            { "from": "fan", "to": "det" },
            { "from": "det", "to": "pkts" }
        ],
        "outputs": { "pkts": { "type": "packets", "queue": 4096 } }
    })
}

fn build(spec: &Value) -> Pipeline {
    registry()
        .build_json(&spec.to_string(), &BuildOptions::default())
        .unwrap()
}

fn errors_of(spec: &Value) -> Vec<crate::Diagnostic> {
    registry()
        .validate_json(&spec.to_string(), &BuildOptions::default())
        .errors
}

fn signal(n: usize) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED, 1);
    (0..n)
        .map(|_| C32::new(2.0 * rng.normal() as f32, 2.0 * rng.normal() as f32))
        .collect()
}

/// The packets `(index, lane, width)` of `x`, computed outside the pipeline; the index counts
/// steps from `origin`.
fn reference(x: &[C32], width: usize, min_power: f64, origin: u64) -> Vec<(u64, u8, u8)> {
    let mut state = BurstState::new(min_power);
    let mut lanes = vec![C32::default(); width];
    let mut out = Vec::new();
    for pair in x.chunks_exact(2) {
        for (k, v) in lanes.iter_mut().enumerate() {
            *v = fan_value(pair[0], pair[1], k, width);
        }
        let index = origin + state.steps;
        if let Some(k) = state.step(&lanes) {
            out.push((index, k as u8, width as u8));
        }
    }
    out
}

fn triples(packets: &[Packet]) -> Vec<(u64, u8, u8)> {
    packets
        .iter()
        .map(|p| (p.index, p.data()[0], p.data()[1]))
        .collect()
}

/// Pushes `x` in the chunks of `sizes` and reads the packets after each push.
fn run(p: &mut Pipeline, x: &[C32], sizes: &mut dyn Iterator<Item = usize>) -> Vec<Packet> {
    let mut got = Vec::new();
    let mut buf = vec![Packet::default(); 4096];
    let mut at = 0;
    while at < x.len() {
        let n = sizes.next().unwrap().min(x.len() - at);
        p.push("iq", InputData::C32(&x[at..at + n])).unwrap();
        at += n;
        let k = p.read_packets("pkts", &mut buf).unwrap();
        got.extend_from_slice(&buf[..k]);
    }
    got
}

// ------------------------------------------------------------------ tests

#[test]
fn a_lane_input_on_a_packet_decoder_validates_builds_and_runs() {
    let spec = flow(16);
    let report = registry().validate_json(&spec.to_string(), &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
    let mut p = build(&spec);
    let x = signal(5000);
    let got = run(&mut p, &x, &mut std::iter::repeat(1000));
    let want = reference(&x, 16, MIN_POWER, 0);
    assert!(want.len() > 200, "the detector did work: {}", want.len());
    assert_eq!(triples(&got), want);
    // The index counts steps of the lane input (2500 of them), not input samples.
    assert!(got.iter().all(|q| q.index < 2500));
    assert!(got.last().unwrap().index > 2000);
    assert!(got.iter().all(|q| q.flags == PacketFlags::NONE));
    let stats = p.stats();
    let fan = stats.iter().find(|s| s.name == "fan").unwrap();
    assert_eq!((fan.samples_in, fan.samples_out), (5000, 2500));
    let det = stats.iter().find(|s| s.name == "det").unwrap();
    assert_eq!(det.samples_in, 2500);
    assert_eq!(det.samples_out, got.len() as u64);
}

#[test]
fn the_reader_learns_the_width_of_its_lane_input() {
    let x = signal(2000);
    for width in [1, 3, 8] {
        let got = run(&mut build(&flow(width)), &x, &mut std::iter::repeat(700));
        assert_eq!(triples(&got), reference(&x, width, MIN_POWER, 0), "{width}");
        assert!(got.iter().all(|q| usize::from(q.data()[1]) == width));
    }
}

#[test]
fn the_packet_index_after_a_gap_counts_steps_from_the_new_origin() {
    let mut p = build(&flow(16));
    let x = signal(3000);
    let first = BlockMeta {
        start: SampleIndex(0),
        ..BlockMeta::default()
    };
    p.push_with_meta("iq", InputData::C32(&x[..1000]), &first)
        .unwrap();
    // 2000 samples are missing: the lane stream continues 1000 steps later, at step 1500.
    let second = BlockMeta {
        start: SampleIndex(3000),
        ..BlockMeta::default()
    };
    p.push_with_meta("iq", InputData::C32(&x[1000..]), &second)
        .unwrap();
    let mut buf = vec![Packet::default(); 4096];
    let k = p.read_packets("pkts", &mut buf).unwrap();
    let mut want = reference(&x[..1000], 16, MIN_POWER, 0);
    // The gap resets the detector; its steps count from the origin of the lane stream.
    want.extend(reference(&x[1000..], 16, MIN_POWER, 1500));
    assert_eq!(triples(&buf[..k]), want);
}

#[test]
fn the_whole_graph_is_chunk_invariant_with_20_plans() {
    let x = signal(6001);
    let want = run(&mut build(&flow(16)), &x, &mut std::iter::repeat(4096));
    assert_eq!(triples(&want), reference(&x[..6000], 16, MIN_POWER, 0));
    for k in 0..20 {
        let plan = ChunkPlan::new(SEED, 1 + 200 * k);
        let got = run(&mut build(&flow(16)), &x, &mut plan.sizes());
        assert!(got == want, "chunk plan {k}");
    }
}

#[test]
fn the_runtime_parameter_reaches_the_block() {
    let mut p = build(&flow(16));
    let x = signal(4000);
    let first = run(&mut p, &x[..2000], &mut std::iter::repeat(2000));
    p.set_param("det", "min_power", json!(1e9)).unwrap();
    let second = run(&mut p, &x[2000..], &mut std::iter::repeat(2000));
    assert_eq!(triples(&first), reference(&x[..2000], 16, MIN_POWER, 0));
    assert!(second.is_empty(), "{second:?}");
}

#[test]
fn reset_starts_the_index_at_0_again() {
    let mut p = build(&flow(16));
    let x = signal(2000);
    let first = run(&mut p, &x, &mut std::iter::repeat(2000));
    p.reset();
    let again = run(&mut p, &x, &mut std::iter::repeat(2000));
    assert_eq!(first, again);
}

#[test]
fn the_report_names_the_step_rate_and_the_packet_rate() {
    let spec = flow(16);
    let report = registry().validate_json(&spec.to_string(), &BuildOptions::default());
    let fan = report.nodes.iter().find(|n| n.name == "fan").unwrap();
    assert_eq!(fan.lanes, Some(16));
    assert_eq!(fan.rate_out, Rate::hz(24_000).unwrap());
    let det = report.nodes.iter().find(|n| n.name == "det").unwrap();
    assert_eq!(det.lanes, None);
    assert_eq!(det.rate_in, Rate::hz(24_000).unwrap());
    // The nominal rate of the packet stream: the step rate times `max_per_input`.
    assert_eq!(det.rate_out, Rate::hz(12_000).unwrap());
    assert_eq!(det.clock_domain, "iq");
    let json: Value = serde_json::from_str(&report.to_json()).unwrap();
    let nodes = json["nodes"].as_array().unwrap();
    let det = nodes.iter().find(|n| n["name"] == "det").unwrap();
    assert_eq!(det["rate_in"], 24_000);
    assert_eq!(det["rate_out"], 12_000);
    assert!(det.get("lanes").is_none());
    // The cost is per input element of the node: 3 per lane, 48 per step, 24 per input
    // sample of the pipeline; `lane_fan` adds 16 / 2 per input sample.
    assert!(
        (report.cost_per_input_sample - (8.0 + 24.0)).abs() < 1e-9,
        "{}",
        report.cost_per_input_sample
    );
    // The built pipeline reports the same.
    let p = build(&spec);
    let det = p.report().nodes.iter().find(|n| n.name == "det").unwrap();
    assert_eq!(det.rate_out, Rate::hz(12_000).unwrap());
}

#[test]
fn a_stream_into_the_lane_input_of_a_packet_decoder_is_a_type_mismatch() {
    let spec = json!({
        "sdrk": 1, "name": "bad",
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": { "det": { "kind": "lane_burst" } },
        "edges": [ { "from": "iq", "to": "det" }, { "from": "det", "to": "pkts" } ],
        "outputs": { "pkts": { "type": "packets" } }
    });
    let errors = errors_of(&spec);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].code, DiagCode::TypeMismatch);
    assert_eq!(errors[0].path, "edges[0].to");
    assert!(
        errors[0].message.contains("takes lanes of c32")
            && errors[0].message.contains("delivers c32"),
        "{}",
        errors[0].message
    );
    // The same behind a stream node.
    let spec = json!({
        "sdrk": 1, "name": "bad",
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": {
            "dec": { "kind": "ddc", "params": { "decimation": 2 } },
            "det": { "kind": "lane_burst" }
        },
        "edges": [
            { "from": "iq", "to": "dec" },
            { "from": "dec", "to": "det" },
            { "from": "det", "to": "pkts" }
        ],
        "outputs": { "pkts": { "type": "packets" } }
    });
    let errors = errors_of(&spec);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].code, DiagCode::TypeMismatch);
    assert_eq!(errors[0].path, "edges[1].to");
}

#[test]
fn the_packets_of_a_lane_decoder_go_to_packet_outputs_only() {
    for output_type in ["stream", "frames"] {
        let mut spec = flow(16);
        spec["outputs"]["pkts"] = json!({ "type": output_type });
        let errors = errors_of(&spec);
        assert_eq!(errors.len(), 1, "{output_type}: {errors:?}");
        assert_eq!(errors[0].code, DiagCode::TypeMismatch);
        assert_eq!(errors[0].path, "outputs.pkts.type");
    }
    // A stream block cannot read the packets either.
    let mut spec = flow(16);
    spec["nodes"]["demod"] = json!({ "kind": "quad_demod" });
    spec["edges"] = json!([
        { "from": "iq", "to": "fan" },
        { "from": "fan", "to": "det" },
        { "from": "det", "to": "demod" },
        { "from": "demod", "to": "pkts" }
    ]);
    spec["outputs"]["pkts"] = json!({ "type": "stream" });
    let errors = errors_of(&spec);
    assert!(
        errors
            .iter()
            .any(|d| d.code == DiagCode::TypeMismatch && d.path == "edges[2].to"),
        "{errors:?}"
    );
}
