//! Lane streams in a pipeline (M45 section 3; the parts of spec test 6 that need no
//! `channelize` kind).
//!
//! The tests register two small test kinds: `lane_spread` (`C32` -> `Lanes<C32>`, one step of
//! `channels` lanes per two input samples, lane `k` of step `i` is `x[2i]·(k+1) + x[2i+1]`,
//! group delay 1/2 input sample) and `lane_pick` (`Lanes<C32>` -> `C32`, copies the lane
//! `lane`, which must be below the width of its input). The rest of test 6 with the kinds
//! `channelize` and `channel_select`, including the counting allocator, is in
//! `tests/channelize_kinds.rs`; the last test here checks their shared lane buffer.
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
    Block, BlockDescription, BlockTiming, LaneElem, Lanes, LanesMut, LanesRef, Mutability,
    ParamDesc, ParamKind, ParamValue, PortDesc, PortType, Unit,
};
use sdrk_testkit::{ChunkPlan, Pcg32};
use serde_json::{Value, json};

use crate::kinds::{Factory, Figures, Kind};
use crate::model::Precision;
use crate::node::{LanesNode, Node};
use crate::registry::{BuildOptions, PortKind, Registry};
use crate::runner::{Buffer, InputData, Pipeline};
use crate::storage::VecStorage;
use crate::validate::DiagCode;

const SEED: u64 = 666;

static SPREAD: BlockDescription = BlockDescription {
    kind: "lane_spread",
    summary: "Test block: one step of lanes per two input samples.",
    details: "Lane k of step i is x[2i]·(k+1) + x[2i+1].",
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

static PICK: BlockDescription = BlockDescription {
    kind: "lane_pick",
    summary: "Test block: copies one lane into a stream.",
    details: "Writes lane `lane` of every step.",
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
        types: &[PortType::C32],
        description: "The lane.",
    }],
    params: &[ParamDesc {
        name: "lane",
        unit: Unit::None,
        kind: ParamKind::I64 {
            min: 0,
            max: 255,
            default: None,
        },
        typical: "0",
        mutability: Mutability::Build,
        description: "The lane to copy, below the width of the input.",
    }],
    required: &["lane"],
};

fn no_param() -> sdrk_blocks::Error {
    sdrk_blocks::Error::UnknownParameter
}

fn timing(ratio: Ratio, delay: Delay) -> BlockTiming {
    BlockTiming {
        ratio,
        kind: RatioKind::Constant,
        delay,
    }
}

/// The value of lane `k` of a step from the input pair `(a, b)`.
fn spread_value(a: C32, b: C32, k: usize) -> C32 {
    a * (k as f32 + 1.0) + b
}

#[derive(Debug)]
struct Spread {
    width: usize,
}

impl Block for Spread {
    type In = C32;
    type Out = Lanes<C32>;

    fn description(&self) -> &'static BlockDescription {
        &SPREAD
    }

    fn process(
        &mut self,
        input: &[C32],
        mut output: LanesMut<'_, C32>,
    ) -> sdrk_blocks::Result<Progress> {
        let steps = (input.len() / 2).min(output.steps());
        for (i, pair) in input.chunks_exact(2).take(steps).enumerate() {
            let step = output.step_mut(i).ok_or(no_param())?;
            for (k, v) in step.iter_mut().enumerate() {
                *v = spread_value(pair[0], pair[1], k);
            }
        }
        Ok(Progress {
            consumed: 2 * steps,
            produced: steps,
        })
    }

    fn timing(&self) -> BlockTiming {
        timing(Ratio::new(1, 2).unwrap(), Delay::new(1, 2).unwrap())
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

#[derive(Debug)]
struct Pick {
    lane: usize,
}

impl Block for Pick {
    type In = Lanes<C32>;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &PICK
    }

    fn process(
        &mut self,
        input: LanesRef<'_, C32>,
        output: &mut [C32],
    ) -> sdrk_blocks::Result<Progress> {
        let n = input.steps().min(output.len());
        for (i, o) in output.iter_mut().take(n).enumerate() {
            *o = *input
                .step(i)
                .and_then(|s| s.get(self.lane))
                .ok_or(no_param())?;
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        timing(Ratio::ONE, Delay::ZERO)
    }

    fn reset(&mut self) {}

    fn set_param(&mut self, _name: &str, _value: ParamValue) -> sdrk_blocks::Result<()> {
        Err(no_param())
    }

    fn param(&self, name: &str) -> sdrk_blocks::Result<ParamValue> {
        match name {
            "lane" => Ok(ParamValue::I64(self.lane as i64)),
            _ => Err(no_param()),
        }
    }

    fn min_input(&self) -> usize {
        1
    }
}

struct SpreadKind;

impl Kind for SpreadKind {
    type Cfg = usize;

    const KIND: &'static str = "lane_spread";
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
            delay: Delay::new(1, 2)?,
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
        Ok(Box::new(LanesNode::new(Spread { width: *cfg })))
    }
}

/// The lane and the width of the input.
#[derive(Debug, Clone, Copy)]
struct PickCfg {
    lane: usize,
    width: usize,
}

struct PickKind;

impl Kind for PickKind {
    type Cfg = PickCfg;

    const KIND: &'static str = "lane_pick";
    fn kernels() -> &'static [&'static str] {
        &[]
    }
    fn default_cfg() -> PickCfg {
        PickCfg { lane: 0, width: 0 }
    }
    fn set(cfg: &mut PickCfg, name: &str, value: ParamValue) -> sdrk_blocks::Result<()> {
        match name {
            "lane" => {
                cfg.lane = usize::try_from(value.as_i64()?).map_err(|_| no_param())?;
                Ok(())
            }
            _ => Err(no_param()),
        }
    }
    fn input_width(cfg: &mut PickCfg, width: usize) -> sdrk_blocks::Result<()> {
        cfg.width = width;
        Ok(())
    }
    fn plan(
        cfg: &PickCfg,
        input: PortType,
        rate: Rate,
        _p: Precision,
    ) -> sdrk_blocks::Result<Figures> {
        assert!(input.is_lanes(), "the factory passes the lane port type");
        if cfg.lane >= cfg.width {
            return Err(sdrk_blocks::Error::OutOfRange {
                name: "lane",
                min: 0.0,
                max: cfg.width.saturating_sub(1) as f64,
                got: cfg.lane as f64,
            });
        }
        Ok(Figures {
            rate_out: rate,
            memory_bytes: 0,
            cost: 1.0,
            taps: 0,
            delay: Delay::ZERO,
            min_input: 1,
            output: PortKind::Stream(PortType::C32),
        })
    }
    fn build(
        cfg: &PickCfg,
        _input: PortType,
        _rate: Rate,
        _p: Precision,
        _st: &mut VecStorage,
    ) -> sdrk_blocks::Result<Box<dyn Node>> {
        Ok(Box::new(LanesNode::new(Pick { lane: cfg.lane })))
    }
}

fn registry() -> Registry {
    let mut r = Registry::standard();
    r.insert_factory(Factory::<SpreadKind>::boxed(&SPREAD));
    r.insert_factory(Factory::<PickKind>::boxed(&PICK));
    r
}

/// The lanes of the four readers.
const LANES: [usize; 4] = [0, 5, 15, 5];

/// A `lane_spread` node with 16 lanes and four `lane_pick` readers (lanes 0, 5, 15 and 5
/// again), each into its own output.
fn flow() -> Value {
    let mut nodes = serde_json::Map::new();
    nodes.insert(
        "chan".into(),
        json!({ "kind": "lane_spread", "params": { "channels": 16 } }),
    );
    let mut edges = vec![json!({ "from": "iq", "to": "chan" })];
    let mut outputs = serde_json::Map::new();
    for (i, lane) in LANES.iter().enumerate() {
        let sel = format!("sel{i}");
        let out = format!("out{i}");
        nodes.insert(
            sel.clone(),
            json!({ "kind": "lane_pick", "params": { "lane": lane } }),
        );
        edges.push(json!({ "from": "chan", "to": sel }));
        edges.push(json!({ "from": sel, "to": out }));
        outputs.insert(out, json!({ "type": "stream" }));
    }
    json!({
        "sdrk": 1, "name": "lanes",
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": nodes,
        "edges": edges,
        "outputs": outputs
    })
}

fn build(spec: &Value) -> Pipeline {
    registry()
        .build_json(&spec.to_string(), &BuildOptions::default())
        .unwrap()
}

fn signal(n: usize) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED, 1);
    (0..n)
        .map(|_| C32::new(rng.normal() as f32, rng.normal() as f32))
        .collect()
}

/// The output of each reader, computed outside the pipeline.
fn reference(x: &[C32]) -> Vec<Vec<C32>> {
    LANES
        .iter()
        .map(|&k| {
            x.chunks_exact(2)
                .map(|p| spread_value(p[0], p[1], k))
                .collect()
        })
        .collect()
}

/// Pushes `x` in the chunks of `sizes` and reads every output after each push.
fn run(p: &mut Pipeline, x: &[C32], sizes: &mut dyn Iterator<Item = usize>) -> Vec<Vec<C32>> {
    let mut got = vec![Vec::new(); LANES.len()];
    let mut buf = vec![C32::default(); 8192];
    let mut at = 0;
    while at < x.len() {
        let n = sizes.next().unwrap().min(x.len() - at);
        p.push("iq", InputData::C32(&x[at..at + n])).unwrap();
        at += n;
        for (i, g) in got.iter_mut().enumerate() {
            let k = p.read_stream(&format!("out{i}"), &mut buf).unwrap();
            g.extend_from_slice(&buf[..k]);
        }
    }
    got
}

/// The lane buffers of a pipeline: (width, steps, readers, data pointer).
fn lane_buffers(p: &Pipeline) -> Vec<(usize, usize, usize, *const C32)> {
    p.streams
        .iter()
        .filter_map(|s| match &s.buf {
            Buffer::Lanes { data, width } => {
                Some((*width, s.buf.len(), s.readers.len(), data.as_ptr()))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn four_readers_share_one_lane_buffer_and_get_their_lanes() {
    let spec = flow();
    let report = registry().validate_json(&spec.to_string(), &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
    let mut p = build(&spec);
    // One lane buffer of 16 lanes, read by all four readers without a copy; its capacity
    // counts steps.
    let buffers = lane_buffers(&p);
    assert_eq!(buffers.len(), 1);
    let (width, steps, readers, data) = buffers[0];
    assert_eq!((width, readers), (16, 4));
    assert!(steps >= 4096 / 2, "{steps} steps");
    let x = signal(5000);
    let got = run(&mut p, &x, &mut std::iter::repeat(1000));
    let want = reference(&x);
    assert_eq!(got, want);
    // Nothing was reallocated while running (the counting allocator of test 6 is in
    // `tests/channelize_kinds.rs`).
    assert_eq!(lane_buffers(&p)[0].3, data);
    let stats = p.stats();
    let chan = stats.iter().find(|s| s.name == "chan").unwrap();
    assert_eq!((chan.samples_in, chan.samples_out), (5000, 2500));
    let sel = stats.iter().find(|s| s.name == "sel2").unwrap();
    assert_eq!((sel.samples_in, sel.samples_out), (2500, 2500));
}

#[test]
fn the_whole_graph_is_chunk_invariant_with_20_plans() {
    let x = signal(6001);
    let want = run(&mut build(&flow()), &x, &mut std::iter::repeat(4096));
    assert_eq!(want, reference(&x[..6000]));
    for k in 0..20 {
        let plan = ChunkPlan::new(SEED, 1 + 200 * k);
        let got = run(&mut build(&flow()), &x, &mut plan.sizes());
        assert!(got == want, "chunk plan {k}");
    }
}

#[test]
fn the_report_names_the_width_and_the_step_rate() {
    let spec = flow();
    let report = registry().validate_json(&spec.to_string(), &BuildOptions::default());
    let chan = report.nodes.iter().find(|n| n.name == "chan").unwrap();
    assert_eq!(chan.lanes, Some(16));
    assert_eq!(chan.rate_out, Rate::hz(24_000).unwrap());
    assert_eq!(chan.clock_domain, "iq");
    let sel = report.nodes.iter().find(|n| n.name == "sel1").unwrap();
    assert_eq!(sel.lanes, None);
    assert_eq!(sel.rate_in, Rate::hz(24_000).unwrap());
    assert_eq!(sel.rate_out, Rate::hz(24_000).unwrap());
    // A lane stream has the clock domain of its source.
    assert_eq!(sel.clock_domain, "iq");
    let json: Value = serde_json::from_str(&report.to_json()).unwrap();
    let nodes = json["nodes"].as_array().unwrap();
    let entry = |name: &str| nodes.iter().find(|n| n["name"] == name).unwrap().clone();
    assert_eq!(entry("chan")["lanes"], 16);
    assert_eq!(entry("chan")["rate_out"], 24_000);
    assert!(entry("sel0").get("lanes").is_none());
    // The delay of the producer (1/2 input sample) is 1/4 step behind it; the outputs count
    // it in their samples, which are steps.
    for o in &report.outputs {
        assert_eq!(o.group_delay, Delay::new(1, 4).unwrap(), "{}", o.name);
    }
    // The built pipeline reports the same.
    let p = build(&spec);
    let chan = p.report().nodes.iter().find(|n| n.name == "chan").unwrap();
    assert_eq!(chan.lanes, Some(16));
}

fn errors_of(spec: &Value) -> Vec<crate::Diagnostic> {
    registry()
        .validate_json(&spec.to_string(), &BuildOptions::default())
        .errors
}

#[test]
fn a_lane_stream_into_a_stream_port_is_a_type_mismatch() {
    let spec = json!({
        "sdrk": 1, "name": "bad",
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": {
            "chan": { "kind": "lane_spread", "params": { "channels": 16 } },
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
fn a_lane_stream_into_port_b_of_a_two_input_node_is_a_type_mismatch() {
    let spec = json!({
        "sdrk": 1, "name": "bad",
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": {
            "chan": { "kind": "lane_spread", "params": { "channels": 16 } },
            "half": { "kind": "decimate", "params": { "factor": 2 } },
            "sum": { "kind": "add" }
        },
        "edges": [
            { "from": "iq", "to": "chan" },
            { "from": "iq", "to": "half" },
            { "from": "half", "to": "sum" },
            { "from": "chan", "to": "sum", "port": "b" },
            { "from": "sum", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    let errors = errors_of(&spec);
    assert!(
        errors.iter().any(|d| d.code == DiagCode::TypeMismatch
            && d.path == "edges[3].to"
            && d.message.contains("lanes of c32, width 16")),
        "{errors:?}"
    );
}

#[test]
fn a_stream_into_a_lane_port_is_a_type_mismatch() {
    let spec = json!({
        "sdrk": 1, "name": "bad",
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": {
            "dec": { "kind": "ddc", "params": { "decimation": 2 } },
            "sel": { "kind": "lane_pick", "params": { "lane": 0 } }
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
        errors[0].message.contains("takes lanes of c32")
            && errors[0].message.contains("delivers c32"),
        "{}",
        errors[0].message
    );
}

#[test]
fn a_lane_stream_into_an_output_is_a_type_mismatch() {
    for output_type in ["stream", "frames", "packets"] {
        let spec = json!({
            "sdrk": 1, "name": "bad",
            "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
            "nodes": { "chan": { "kind": "lane_spread", "params": { "channels": 16 } } },
            "edges": [ { "from": "iq", "to": "chan" }, { "from": "chan", "to": "all" } ],
            "outputs": { "all": { "type": output_type } }
        });
        let errors = errors_of(&spec);
        assert_eq!(errors.len(), 1, "{output_type}: {errors:?}");
        assert_eq!(errors[0].code, DiagCode::TypeMismatch);
        assert_eq!(errors[0].path, "outputs.all.type");
        assert!(
            errors[0].message.contains("lanes of c32, width 16"),
            "{}",
            errors[0].message
        );
    }
}

#[test]
fn a_reader_learns_the_width_of_its_input() {
    let mut spec = flow();
    spec["nodes"]["sel2"]["params"]["lane"] = json!(16);
    let errors = errors_of(&spec);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].code, DiagCode::RateUnsupported);
    assert_eq!(errors[0].path, "nodes.sel2.params.lane");
    assert!(
        errors[0].message.contains("0 to 15"),
        "{}",
        errors[0].message
    );
    // With 32 lanes the same lane is valid.
    spec["nodes"]["chan"]["params"]["channels"] = json!(32);
    assert_eq!(errors_of(&spec), Vec::new());
}

#[test]
fn the_lane_buffer_counts_in_the_memory_estimate() {
    let narrow = registry().validate_json(&flow().to_string(), &BuildOptions::default());
    let mut wide = flow();
    wide["nodes"]["chan"]["params"]["channels"] = json!(32);
    let wide = registry().validate_json(&wide.to_string(), &BuildOptions::default());
    // 16 more lanes of 8 bytes for every step of the buffer.
    let steps = lane_buffers(&build(&flow()))[0].1;
    assert_eq!(wide.memory_bytes - narrow.memory_bytes, steps * 16 * 8);
    // The built pipeline counts the same bytes.
    assert_eq!(build(&flow()).report().memory_bytes, narrow.memory_bytes);
}

#[test]
fn reset_empties_the_lane_buffer() {
    let mut p = build(&flow());
    let x = signal(64);
    // An odd count leaves half a step in the input buffer, nothing in the lane buffer.
    let first = run(&mut p, &x[..33], &mut std::iter::repeat(33));
    assert_eq!(first[0].len(), 16);
    p.reset();
    let again = run(&mut p, &x, &mut std::iter::repeat(64));
    assert_eq!(again, reference(&x));
}

#[test]
fn a_lane_edge_cannot_cross_groups() {
    let mut spec = flow();
    spec["groups"] = json!({ "front": {}, "back": {} });
    spec["nodes"]["chan"]["group"] = json!("front");
    for i in 0..LANES.len() {
        spec["nodes"][format!("sel{i}")]["group"] = json!("back");
    }
    let spec = crate::PipelineSpec::from_json(&spec.to_string()).unwrap();
    let report = crate::split_by_groups(&spec, &registry()).unwrap_err();
    assert!(
        report
            .errors
            .iter()
            .any(|d| d.code == DiagCode::InvalidValue
                && d.path == "edges.chan"
                && d.message.contains("lanes")),
        "{}",
        report.to_json()
    );
}

#[test]
fn the_channelize_kind_has_one_lane_buffer_for_four_readers() {
    // Spec test 6 with the standard kinds: `channelize` with four `channel_select` readers.
    let mut spec = flow();
    spec["nodes"]["chan"] = json!({ "kind": "channelize", "params": { "channels": 16 } });
    for (i, channel) in [0, -1, 7, -1].iter().enumerate() {
        spec["nodes"][format!("sel{i}")] =
            json!({ "kind": "channel_select", "params": { "channel": channel } });
    }
    let mut p = Registry::standard()
        .build_json(&spec.to_string(), &BuildOptions::default())
        .unwrap();
    let buffers = lane_buffers(&p);
    assert_eq!(buffers.len(), 1);
    let (width, _, readers, data) = buffers[0];
    assert_eq!((width, readers), (16, 4));
    let x = signal(4000);
    let got = run(&mut p, &x, &mut std::iter::repeat(1000));
    assert!(got.iter().all(|g| g.len() == 500));
    assert_eq!(got[1], got[3]);
    assert_eq!(lane_buffers(&p)[0].3, data);
}
