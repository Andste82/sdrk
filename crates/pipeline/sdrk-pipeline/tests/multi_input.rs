//! Nodes with two inputs (M34): the edge field `port`, the validation of rates, clocks and
//! delays, the automatic alignment, the lockstep runner in push and pull mode, and the proof
//! flow `fm_stereo_receiver_primitives`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::needless_pass_by_value
)]

mod common;

use num_complex::Complex64;
use sdrk::{C32, Q15};
use sdrk_blocks::{AddBlock, ArenaStorage, Block};
use sdrk_device::BlockFlags;
use sdrk_pipeline::{
    BuildOptions, DiagCode, EventKind, InputData, Pipeline, PipelineSpec, PullError, PullSource,
    Registry, split_by_groups,
};
use sdrk_testkit::e2e_stereo::{StereoSignal, StereoSource};
use sdrk_testkit::{ChunkPlan, Pcg32};
use serde_json::{Value, json};

use common::{build, flow};

/// The one seed of M34 (the seed table of the milestones): noise, test signal and chunk plans.
const SEED: u64 = 478;

fn validate(doc: &Value) -> sdrk_pipeline::Report {
    Registry::standard().validate_json(&doc.to_string(), &BuildOptions::default())
}

fn codes(report: &sdrk_pipeline::Report) -> Vec<(String, String)> {
    let mut v: Vec<_> = report
        .errors
        .iter()
        .map(|d| (d.code.as_str().to_string(), d.path.clone()))
        .collect();
    v.sort();
    v
}

// ------------------------------------------------------------------ the proof flow

/// The stereo signal of M24 at the 480 kHz baseband as `cs16` bytes (the station at 0 Hz).
fn stereo_cs16(samples: usize) -> Vec<u8> {
    const CHUNK: usize = 500;
    let down: Value = json!({
        "sdrk": 1, "name": "front", "settings": { "max_chunk": 4096 },
        "inputs": { "rf": { "format": "cu8", "rate_hz": 2_400_000 } },
        "nodes": { "tune": { "kind": "ddc", "params": {
            "cutoff_hz": 130_000, "decimation": 5, "frequency_hz": 600_000, "transition_hz": 30_000
        } } },
        "edges": [ { "from": "rf", "to": "tune" }, { "from": "tune", "to": "iq" } ],
        "outputs": { "iq": { "type": "stream", "capacity": 16384 } }
    });
    let mut front = build(&down.to_string());
    let signal = StereoSignal::STEREO;
    let mut source = StereoSource::new(signal, SEED);
    let mut scratch = vec![Complex64::new(0.0, 0.0); CHUNK];
    let mut bytes = vec![0u8; 2 * CHUNK];
    let mut out = Vec::with_capacity(samples * 4);
    let mut buf = vec![C32::default(); 4096];
    for chunk in 0..samples * 5 / CHUNK {
        source
            .fill(&mut scratch, (chunk * CHUNK) as u64, &mut bytes)
            .unwrap();
        front.push("rf", InputData::Raw(&bytes)).unwrap();
        loop {
            let n = front.read_stream("iq", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            for s in &buf[..n] {
                for v in [s.re, s.im] {
                    let q = (f64::from(v) * 32767.0).round().clamp(-32768.0, 32767.0);
                    out.extend_from_slice(&(q as i16).to_le_bytes());
                }
            }
        }
    }
    out
}

/// Pushes `bytes` (`cs16`) in pieces and returns all audio.
fn run_cs16(p: &mut Pipeline, bytes: &[u8], plan: Option<ChunkPlan>) -> Vec<C32> {
    let mut sizes = plan.map(|p| p.sizes());
    let mut out = Vec::new();
    let mut buf = vec![C32::default(); 4096];
    let mut at = 0;
    while at < bytes.len() {
        let samples = sizes.as_mut().map_or(4096, |s| s.next().unwrap());
        let end = (at + samples * 4).min(bytes.len());
        p.push("rf", InputData::Raw(&bytes[at..end])).unwrap();
        at = end;
        loop {
            let n = p.read_stream("audio", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
        }
    }
    out
}

fn proof_flow() -> Value {
    serde_json::from_str(&flow("fm_stereo_receiver_primitives")).unwrap()
}

/// The proof flow with the primitives from the discriminator to the matrix replaced by one
/// `fm_stereo_demod` node (`deemphasis_us` as given).
fn kind_flow(deemphasis_us: u32) -> Value {
    let mut doc = proof_flow();
    let tune = doc["nodes"]["tune"].clone();
    doc["nodes"] = json!({
        "tune": tune,
        "demod": { "kind": "fm_stereo_demod", "params": {
            "audio_decimation": 5, "deemphasis_us": deemphasis_us
        } }
    });
    doc["edges"] = json!([
        { "from": "rf", "to": "tune" },
        { "from": "tune", "to": "demod" },
        { "from": "demod", "to": "audio" }
    ]);
    doc
}

#[test]
fn the_proof_flow_needs_no_alignment_and_has_a_clock_domain_per_node() {
    let report = validate(&proof_flow());
    assert!(report.is_ok(), "{}", report.to_json());
    assert!(report.warnings.is_empty(), "{}", report.to_json());
    assert!(
        report.nodes.iter().all(|n| !n.name.ends_with("__align")),
        "the band-pass has no group delay in the delay model"
    );
    assert!(report.nodes.iter().all(|n| n.clock_domain == "rf"));
    let pll = report.nodes.iter().find(|n| n.name == "pll").unwrap();
    assert_eq!(pll.kind, "stereo_pll");
    assert_eq!(pll.rate_in.as_f64(), 240_000.0);
}

#[test]
fn fm_stereo_primitives_equal() {
    let bytes = stereo_cs16(120_000);
    let want = run_cs16(&mut build(&kind_flow(0).to_string()), &bytes, None);
    assert!(want.len() > 11_000, "{} samples", want.len());
    let proof = proof_flow().to_string();
    let whole = run_cs16(&mut build(&proof), &bytes, None);
    assert_eq!(whole.len(), want.len());
    assert!(
        whole == want,
        "the flow built from primitives differs from fm_stereo_demod"
    );
    // The same audio with 20 chunk plans (seed 478, largest chunk 150 to 3000 samples).
    for k in 0..20 {
        let plan = ChunkPlan::new(SEED, 150 * (k + 1));
        let got = run_cs16(&mut build(&proof), &bytes, Some(plan));
        assert!(got == want, "chunk plan {k}");
    }
}

#[test]
fn a_de_emphasis_from_primitives_follows_the_kind_closely() {
    // The de-emphasis of `fm_stereo_demod` is a complex biquad section with explicit
    // coefficients; the pipeline has the one-pole kernel `deemphasis` for real streams, so the
    // two channels are split, de-emphasised and joined again with `make_complex` (D388).
    let mut doc = proof_flow();
    let nodes = doc["nodes"].as_object_mut().unwrap();
    nodes.insert("left".into(), json!({ "kind": "real_part" }));
    nodes.insert("right".into(), json!({ "kind": "imag_part" }));
    for side in ["l", "r"] {
        nodes.insert(
            format!("deemph_{side}"),
            json!({ "kind": "deemphasis", "params": { "tau_us": 50 } }),
        );
    }
    nodes.insert("stereo".into(), json!({ "kind": "make_complex" }));
    let edges = doc["edges"].as_array_mut().unwrap();
    edges.retain(|e| e["to"] != "audio");
    edges.extend([
        json!({ "from": "matrix", "to": "left" }),
        json!({ "from": "matrix", "to": "right" }),
        json!({ "from": "left", "to": "deemph_l" }),
        json!({ "from": "right", "to": "deemph_r" }),
        json!({ "from": "deemph_l", "to": "stereo", "port": "a" }),
        json!({ "from": "deemph_r", "to": "stereo", "port": "b" }),
        json!({ "from": "stereo", "to": "audio" }),
    ]);
    let report = validate(&doc);
    assert!(report.is_ok(), "{}", report.to_json());
    assert!(report.warnings.is_empty(), "{}", report.to_json());
    let bytes = stereo_cs16(120_000);
    let want = run_cs16(&mut build(&kind_flow(50).to_string()), &bytes, None);
    let got = run_cs16(&mut build(&doc.to_string()), &bytes, None);
    assert_eq!(got.len(), want.len());
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs()))
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-6, "largest difference {worst}");
    let peak = want.iter().map(|s| s.re.abs()).fold(0.0f32, f32::max);
    assert!(peak > 0.1, "the audio is not silent: {peak}");
}

// ------------------------------------------------------------------ the model

#[test]
fn the_port_survives_a_round_trip_and_is_written_only_when_given() {
    let text = flow("fm_stereo_receiver_primitives");
    let spec = sdrk_pipeline::PipelineSpec::from_json(&text).unwrap();
    let again = sdrk_pipeline::PipelineSpec::from_json(&spec.to_json()).unwrap();
    assert_eq!(spec, again);
    let with_port = spec
        .edges
        .iter()
        .filter(|e| e.port == Some(sdrk_pipeline::PortName::B))
        .count();
    assert_eq!(with_port, 1, "only the edge into port b names its port");
    assert_eq!(
        text.matches("\"port\"").count(),
        2,
        "`a` and `b` of the loop"
    );
    // The configuration of a built pipeline writes the ports as they were given.
    let p = build(&text);
    assert_eq!(p.config(), spec);
    // An explicit `a` is kept and means the same as no port (D381).
    let doc = json!({
        "sdrk": 1, "name": "t", "settings": { "max_chunk": 64 },
        "inputs": { "x": { "type": "f32", "rate_hz": 48000 } },
        "nodes": { "sum": { "kind": "add" } },
        "edges": [
            { "from": "x", "to": "sum", "port": "a" },
            { "from": "x", "to": "sum", "port": "b" },
            { "from": "sum", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    let spec = sdrk_pipeline::PipelineSpec::from_json(&doc.to_string()).unwrap();
    assert_eq!(spec.edges[0].port_or_default(), sdrk_pipeline::PortName::A);
    assert_eq!(
        serde_json::from_str::<Value>(&spec.to_json()).unwrap()["edges"],
        doc["edges"],
        "the edges are unchanged"
    );
}

#[test]
fn port_and_align_are_checked_in_the_structure_stage() {
    let doc = json!({
        "sdrk": 1, "name": "t",
        "inputs": { "x": { "type": "f32", "rate_hz": 48000 } },
        "nodes": { "sum": { "kind": "add", "align": "sometimes" } },
        "edges": [
            { "from": "x", "to": "sum", "port": "c" },
            { "from": "x", "to": "sum", "port": "b", "extra": 1 },
            { "from": "sum", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    let report = validate(&doc);
    assert_eq!(
        codes(&report),
        [
            ("invalid_value".to_string(), "edges[0].port".to_string()),
            ("invalid_value".to_string(), "nodes.sum.align".to_string()),
            ("unknown_field".to_string(), "edges[1].extra".to_string()),
        ]
    );
}

// ------------------------------------------------------------------ validation

/// Two paths from `x` into `sum`: the direct one and one through `via` (a list of nodes).
fn two_paths(ty: &str, via: Vec<(&str, Value)>, sum: Value) -> Value {
    let mut nodes = serde_json::Map::new();
    nodes.insert("sum".into(), sum);
    let mut edges = vec![json!({ "from": "x", "to": "sum", "port": "a" })];
    let mut last = "x".to_string();
    for (name, node) in via {
        nodes.insert(name.into(), node);
        edges.push(json!({ "from": last, "to": name }));
        last = name.to_string();
    }
    edges.push(json!({ "from": last, "to": "sum", "port": "b" }));
    edges.push(json!({ "from": "sum", "to": "out" }));
    json!({
        "sdrk": 1, "name": "t", "settings": { "max_chunk": 4096 },
        "inputs": { "x": { "type": ty, "rate_hz": 48000 } },
        "nodes": nodes, "edges": edges,
        "outputs": { "out": { "type": "stream" } }
    })
}

#[test]
fn a_missing_port_and_a_second_edge_into_a_port_are_reported_per_port() {
    let mut doc = two_paths("f32", vec![], json!({ "kind": "add" }));
    // Remove the edge into port b.
    doc["edges"].as_array_mut().unwrap().remove(1);
    let report = validate(&doc);
    assert_eq!(
        codes(&report),
        [("missing_input".to_string(), "nodes.sum.ports.b".to_string())]
    );
    // Two edges into port b.
    let mut doc = two_paths("f32", vec![], json!({ "kind": "add" }));
    doc["edges"]
        .as_array_mut()
        .unwrap()
        .insert(2, json!({ "from": "x", "to": "sum", "port": "b" }));
    let report = validate(&doc);
    assert_eq!(
        codes(&report),
        [("multiple_inputs".to_string(), "edges[2].port".to_string())]
    );
    // A second edge into port a keeps the path `edges[k].to` of the one-input case.
    let mut doc = two_paths("f32", vec![], json!({ "kind": "add" }));
    doc["edges"]
        .as_array_mut()
        .unwrap()
        .insert(2, json!({ "from": "x", "to": "sum" }));
    let report = validate(&doc);
    assert_eq!(
        codes(&report),
        [("multiple_inputs".to_string(), "edges[2].to".to_string())]
    );
}

#[test]
fn a_port_on_an_output_or_a_one_input_node_is_unknown() {
    let mut doc = two_paths("f32", vec![], json!({ "kind": "add" }));
    doc["edges"][2] = json!({ "from": "sum", "to": "out", "port": "a" });
    let report = validate(&doc);
    assert_eq!(
        codes(&report),
        [("unknown_port".to_string(), "edges[2].port".to_string())]
    );
    assert!(report.errors[0].message.contains("output"));
    // Port b of a node with one input.
    let doc = json!({
        "sdrk": 1, "name": "t",
        "inputs": { "x": { "type": "f32", "rate_hz": 48000 } },
        "nodes": { "g": { "kind": "scale" } },
        "edges": [ { "from": "x", "to": "g", "port": "b" }, { "from": "g", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    });
    let report = validate(&doc);
    assert_eq!(
        codes(&report),
        [("unknown_port".to_string(), "edges[0].port".to_string())]
    );
    assert!(report.errors[0].message.contains("no input port `b`"));
}

#[test]
fn the_ports_take_the_same_type() {
    // f32 on port a, c32 on port b: the second path goes through `to_complex`.
    let doc = two_paths(
        "f32",
        vec![("cx", json!({ "kind": "to_complex" }))],
        json!({ "kind": "add" }),
    );
    let report = validate(&doc);
    assert_eq!(
        codes(&report),
        [("type_mismatch".to_string(), "edges[2].to".to_string())]
    );
    // `make_complex` takes real streams only; a complex stream on a port is refused at its edge.
    let doc = two_paths(
        "c32",
        vec![("mag", json!({ "kind": "mag" }))],
        json!({ "kind": "make_complex" }),
    );
    let report = validate(&doc);
    assert_eq!(
        codes(&report),
        [("type_mismatch".to_string(), "edges[0].to".to_string())]
    );
}

#[test]
fn different_rates_are_a_rate_mismatch_that_names_both_rates() {
    let doc = two_paths(
        "f32",
        vec![(
            "half",
            json!({ "kind": "decimate", "params": { "factor": 2 } }),
        )],
        json!({ "kind": "add" }),
    );
    let report = validate(&doc);
    assert_eq!(
        codes(&report),
        [("rate_mismatch".to_string(), "edges[2].to".to_string())]
    );
    let message = &report.errors[0].message;
    assert!(
        message.contains("48000") && message.contains("24000"),
        "{message}"
    );
}

#[test]
fn two_inputs_are_two_clocks_even_at_the_same_rate() {
    let doc = json!({
        "sdrk": 1, "name": "t",
        "inputs": {
            "rf": { "type": "f32", "rate_hz": 48000 },
            "audio": { "type": "f32", "rate_hz": 48000 }
        },
        "nodes": { "sum": { "kind": "add" } },
        "edges": [
            { "from": "rf", "to": "sum", "port": "a" },
            { "from": "audio", "to": "sum", "port": "b" },
            { "from": "sum", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    let report = validate(&doc);
    assert_eq!(
        codes(&report),
        [("clock_domain".to_string(), "edges[1].to".to_string())]
    );
    let message = &report.errors[0].message;
    assert!(
        message.contains("`audio`") && message.contains("`rf`"),
        "{message}"
    );
    assert!(message.contains("drift controller"), "{message}");
    // Different rates give both diagnostics.
    let mut other = doc.clone();
    other["inputs"]["audio"]["rate_hz"] = json!(44100);
    assert_eq!(
        codes(&validate(&other)),
        [
            ("clock_domain".to_string(), "edges[1].to".to_string()),
            ("rate_mismatch".to_string(), "edges[1].to".to_string())
        ]
    );
}

/// A low-pass `fir` with exactly 31 taps at 48 kHz (a delay of 15 samples).
fn fir31() -> Value {
    json!({ "kind": "fir", "params": {
        "cutoff_hz": 4000, "transition_hz": 2500, "attenuation_db": 30
    } })
}

#[test]
fn a_filter_on_one_path_gets_an_automatic_delay_on_the_other() {
    let doc = two_paths("c32", vec![("lp", fir31())], json!({ "kind": "add" }));
    let report = validate(&doc);
    assert!(report.is_ok(), "{}", report.to_json());
    let align = report
        .nodes
        .iter()
        .find(|n| n.name == "sum__align")
        .unwrap();
    assert!(align.auto);
    assert_eq!(align.kind, "delay");
    assert_eq!(
        align.delay_samples_in.num() / align.delay_samples_in.den(),
        15
    );
    assert_eq!(align.clock_domain, "x");
    // The delay node sits in front of the node, after the filter's slower path begins.
    let names: Vec<_> = report.nodes.iter().map(|n| n.name.as_str()).collect();
    let (a, s) = (
        names.iter().position(|n| *n == "sum__align").unwrap(),
        names.iter().position(|n| *n == "sum").unwrap(),
    );
    assert!(a < s);
    // The built pipeline carries the delay node, and its figures agree with the plan.
    let p = build(&doc.to_string());
    assert!(
        p.report()
            .nodes
            .iter()
            .any(|n| n.name == "sum__align" && n.auto)
    );
    assert_eq!(
        p.stats().iter().filter(|s| s.name == "sum__align").count(),
        1
    );
    // The automatic node does not appear in the configuration.
    assert!(!p.config().nodes.contains_key("sum__align"));
    // The delay shows in the figures of the output: the same group delay as the filter.
    assert_eq!(
        report.outputs[0].group_delay.num() / report.outputs[0].group_delay.den(),
        15
    );
}

#[test]
fn align_none_inserts_nothing_and_warns() {
    let mut sum = json!({ "kind": "add" });
    sum["align"] = json!("none");
    let doc = two_paths("c32", vec![("lp", fir31())], sum);
    let report = validate(&doc);
    assert!(report.is_ok(), "{}", report.to_json());
    assert!(report.nodes.iter().all(|n| !n.name.ends_with("__align")));
    assert_eq!(report.warnings.len(), 1);
    assert_eq!(report.warnings[0].code, DiagCode::MisalignedInputs);
    assert_eq!(report.warnings[0].path, "nodes.sum");
    assert!(
        report.warnings[0].message.contains("15"),
        "{}",
        report.warnings[0].message
    );
    // The option is part of the configuration.
    let p = build(&doc.to_string());
    assert_eq!(
        p.config().nodes["sum"].align,
        Some(sdrk_pipeline::AlignMode::None)
    );
}

#[test]
fn a_difference_that_is_not_a_whole_number_is_an_error_unless_align_is_none() {
    let via = |sum: Value| {
        two_paths(
            "f32",
            vec![
                (
                    "out_64k",
                    json!({ "kind": "resample", "params": { "rate_out_hz": 64000 } }),
                ),
                (
                    "back",
                    json!({ "kind": "resample", "params": { "rate_out_hz": 48000 } }),
                ),
            ],
            sum,
        )
    };
    let report = validate(&via(json!({ "kind": "add" })));
    assert_eq!(
        codes(&report),
        [("misaligned_inputs".to_string(), "nodes.sum".to_string())]
    );
    assert!(
        report.errors[0].message.contains("73/2"),
        "{}",
        report.errors[0].message
    );
    let report = validate(&via(json!({ "kind": "add", "align": "none" })));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.warnings.len(), 1);
}

#[test]
fn the_automatic_delay_has_a_limit_and_a_reserved_name() {
    // A delay of more than 65536 samples cannot be inserted.
    let doc = two_paths(
        "f32",
        vec![
            (
                "long",
                json!({ "kind": "delay", "params": { "samples": 65536 } }),
            ),
            (
                "longer",
                json!({ "kind": "delay", "params": { "samples": 65536 } }),
            ),
        ],
        json!({ "kind": "add" }),
    );
    let report = validate(&doc);
    assert_eq!(
        codes(&report),
        [("misaligned_inputs".to_string(), "nodes.sum".to_string())]
    );
    // `<node>__align` is taken by the node `sum`.
    let mut doc = two_paths("c32", vec![("lp", fir31())], json!({ "kind": "add" }));
    doc["nodes"]["sum__align"] = json!({ "kind": "scale" });
    doc["edges"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "from": "sum", "to": "sum__align" }));
    doc["edges"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "from": "sum__align", "to": "out2" }));
    doc["outputs"]["out2"] = json!({ "type": "stream" });
    let report = validate(&doc);
    assert!(
        report
            .errors
            .iter()
            .any(|e| e.code == DiagCode::DuplicateName),
        "{}",
        report.to_json()
    );
}

#[test]
fn a_cycle_through_a_second_port_is_found() {
    let doc = json!({
        "sdrk": 1, "name": "t",
        "inputs": { "x": { "type": "f32", "rate_hz": 48000 } },
        "nodes": { "sum": { "kind": "add" }, "g": { "kind": "scale" } },
        "edges": [
            { "from": "x", "to": "sum", "port": "a" },
            { "from": "g", "to": "sum", "port": "b" },
            { "from": "sum", "to": "g" },
            { "from": "sum", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    let report = validate(&doc);
    assert_eq!(report.errors.len(), 1, "{}", report.to_json());
    assert_eq!(report.errors[0].code, DiagCode::Cycle);
}

#[test]
fn both_ports_may_read_the_same_node() {
    let doc = two_paths("c32", vec![], json!({ "kind": "mul" }));
    let report = validate(&doc);
    assert!(report.is_ok(), "{}", report.to_json());
    let mut p = build(&doc.to_string());
    // x * x for complex samples.
    p.push(
        "x",
        InputData::C32(&[C32::new(1.0, 2.0), C32::new(0.0, 3.0)]),
    )
    .unwrap();
    let mut out = [C32::default(); 2];
    assert_eq!(p.read_stream("out", &mut out).unwrap(), 2);
    assert_eq!(out, [C32::new(-3.0, 4.0), C32::new(-9.0, 0.0)]);
}

#[test]
fn the_four_kinds_follow_the_cost_table_and_the_precision() {
    let registry = Registry::standard();
    for (kind, input, cost) in [
        ("add", "c32", 1.0),
        ("add", "f32", 1.0),
        ("mul", "c32", 4.0),
        ("mul", "f32", 1.0),
        ("make_complex", "f32", 1.0),
        ("stereo_pll", "f32", 12.0),
    ] {
        let rate = if kind == "stereo_pll" {
            192_000
        } else {
            48_000
        };
        let doc = json!({
            "sdrk": 1, "name": "t", "settings": { "max_chunk": 64 },
            "inputs": { "x": { "type": input, "rate_hz": rate } },
            "nodes": { "n": { "kind": kind } },
            "edges": [
                { "from": "x", "to": "n", "port": "a" },
                { "from": "x", "to": "n", "port": "b" },
                { "from": "n", "to": "out" }
            ],
            "outputs": { "out": { "type": "stream" } }
        });
        let report = registry.validate_json(&doc.to_string(), &BuildOptions::default());
        assert!(report.is_ok(), "{kind}: {}", report.to_json());
        assert_eq!(report.cost_per_input_sample, cost, "{kind}");
        assert_eq!(
            report
                .nodes
                .iter()
                .find(|n| n.name == "n")
                .unwrap()
                .delay_samples_in
                .num(),
            0
        );
    }
    // Q15 builds the Q15 blocks.
    let doc = json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "q15", "max_chunk": 64 },
        "inputs": { "x": { "type": "q15", "rate_hz": 48000 } },
        "nodes": { "n": { "kind": "add" } },
        "edges": [
            { "from": "x", "to": "n", "port": "a" },
            { "from": "x", "to": "n", "port": "b" },
            { "from": "n", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    let mut p = build(&doc.to_string());
    p.push("x", InputData::Q15(&[Q15(20_000), Q15(-5)]))
        .unwrap();
    let mut out = [Q15(0); 2];
    assert_eq!(p.read_stream("out", &mut out).unwrap(), 2);
    assert_eq!(out, [Q15(32_767), Q15(-10)]);
}

// ------------------------------------------------------------------ the runner

fn noise(n: usize, stream: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED, stream);
    (0..n)
        .map(|_| C32::new(rng.normal() as f32 * 0.25, rng.normal() as f32 * 0.25))
        .collect()
}

/// `x` direct into port a and through a scale into port b of an `add`.
fn add_flow(extra: Vec<(&str, Value)>) -> Value {
    two_paths(
        "c32",
        [
            vec![("g", json!({ "kind": "scale", "params": { "gain_db": -6 } }))],
            extra,
        ]
        .concat(),
        json!({ "kind": "add" }),
    )
}

/// Pushes complex samples in pieces of the given sizes and returns the whole `out` stream.
fn push_all(p: &mut Pipeline, data: &[C32], sizes: &mut dyn Iterator<Item = usize>) -> Vec<C32> {
    let mut out = Vec::new();
    let mut buf = vec![C32::default(); 8192];
    let mut at = 0;
    while at < data.len() {
        let n = sizes.next().unwrap().min(data.len() - at);
        p.push("x", InputData::C32(&data[at..at + n])).unwrap();
        at += n;
        loop {
            let got = p.read_stream("out", &mut buf).unwrap();
            if got == 0 {
                break;
            }
            out.extend_from_slice(&buf[..got]);
        }
    }
    out
}

#[test]
fn add_of_two_paths_equals_the_kernel_on_the_same_samples() {
    let x = noise(20_000, 1);
    let scale_only = json!({
        "sdrk": 1, "name": "t", "settings": { "max_chunk": 4096 },
        "inputs": { "x": { "type": "c32", "rate_hz": 48000 } },
        "nodes": { "g": { "kind": "scale", "params": { "gain_db": -6 } } },
        "edges": [ { "from": "x", "to": "g" }, { "from": "g", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    });
    let y = push_all(
        &mut build(&scale_only.to_string()),
        &x,
        &mut std::iter::repeat(4096),
    );
    assert_eq!(y.len(), x.len());
    let mut want = vec![C32::default(); x.len()];
    let mut arena = ArenaStorage::new(&mut []);
    let mut block = AddBlock::<C32, _>::new(sdrk::Rate::hz(48_000).unwrap(), &mut arena).unwrap();
    block.process((&x, &y), &mut want).unwrap();
    let got = push_all(
        &mut build(&add_flow(vec![]).to_string()),
        &x,
        &mut std::iter::repeat(4096),
    );
    assert!(got == want, "the pipeline differs from the kernel");
}

#[test]
fn the_output_does_not_depend_on_the_chunk_sizes_with_an_automatic_delay() {
    let x = noise(30_000, 2);
    let doc = two_paths("c32", vec![("lp", fir31())], json!({ "kind": "add" }));
    let mut reference = Vec::new();
    for (name, mut sizes) in [
        (
            "4096",
            Box::new(std::iter::repeat(4096usize)) as Box<dyn Iterator<Item = usize>>,
        ),
        ("1", Box::new(std::iter::repeat(1usize))),
        ("7", Box::new(std::iter::repeat(7usize))),
        ("plan", Box::new(ChunkPlan::new(SEED, 3000).sizes())),
    ] {
        let got = push_all(&mut build(&doc.to_string()), &x, &mut sizes);
        assert_eq!(got.len(), x.len(), "{name}");
        if reference.is_empty() {
            reference = got;
        } else {
            assert!(got == reference, "chunk sizes {name}");
        }
    }
    // The output is the sum of the delayed input and the filtered input.
    let delayed = json!({
        "sdrk": 1, "name": "t", "settings": { "max_chunk": 4096 },
        "inputs": { "x": { "type": "c32", "rate_hz": 48000 } },
        "nodes": { "d": { "kind": "delay", "params": { "samples": 15 } } },
        "edges": [ { "from": "x", "to": "d" }, { "from": "d", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    });
    let a = push_all(
        &mut build(&delayed.to_string()),
        &x,
        &mut std::iter::repeat(4096),
    );
    let filtered = json!({
        "sdrk": 1, "name": "t", "settings": { "max_chunk": 4096 },
        "inputs": { "x": { "type": "c32", "rate_hz": 48000 } },
        "nodes": { "lp": fir31() },
        "edges": [ { "from": "x", "to": "lp" }, { "from": "lp", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    });
    let b = push_all(
        &mut build(&filtered.to_string()),
        &x,
        &mut std::iter::repeat(4096),
    );
    let mut want = vec![C32::default(); x.len()];
    let mut arena = ArenaStorage::new(&mut []);
    let mut block = AddBlock::<C32, _>::new(sdrk::Rate::hz(48_000).unwrap(), &mut arena).unwrap();
    block.process((&a, &b), &mut want).unwrap();
    assert!(reference == want, "delay(x, 15) + fir(x)");
    // The first 15 samples of the delayed path are zero: the output starts with the filter.
    assert_eq!(reference[0], b[0]);
}

#[test]
fn a_branch_that_reads_the_input_directly_runs_from_the_buffer() {
    // Both ports read the pipeline input; the node cannot work on the caller's slice.
    let doc = two_paths("c32", vec![], json!({ "kind": "mul" }));
    let x = noise(9_000, 3);
    let whole = push_all(
        &mut build(&doc.to_string()),
        &x,
        &mut std::iter::repeat(4096),
    );
    let single = push_all(&mut build(&doc.to_string()), &x, &mut std::iter::repeat(1));
    assert!(whole == single, "chunk size 1 against one chunk");
    for (s, o) in x.iter().zip(&whole) {
        let want = *s * *s;
        assert!((o.re - want.re).abs() < 1e-6 && (o.im - want.im).abs() < 1e-6);
    }
}

/// An audio chain with two joins: `mix = audio * lowpass(audio)`, then `iq = (mix, lowpass)`,
/// as `cs8` bytes on the output `rf`.
fn pair_transmitter() -> Value {
    json!({
        "sdrk": 1, "name": "pairs", "settings": { "max_chunk": 4800 },
        "inputs": { "audio": { "type": "f32", "rate_hz": 48000 } },
        "nodes": {
            "lp": { "kind": "biquad", "params": { "cutoff_hz": 3000, "response": "lowpass" } },
            "mix": { "kind": "mul" },
            "iq": { "kind": "make_complex" }
        },
        "edges": [
            { "from": "audio", "to": "mix", "port": "a" },
            { "from": "audio", "to": "lp" },
            { "from": "lp", "to": "mix", "port": "b" },
            { "from": "mix", "to": "iq", "port": "a" },
            { "from": "lp", "to": "iq", "port": "b" },
            { "from": "iq", "to": "rf" }
        ],
        "outputs": { "rf": { "type": "stream", "format": "cs8", "underrun": "short", "capacity": 65536 } }
    })
}

struct Audio {
    data: Vec<f32>,
    at: usize,
    asked: usize,
}

impl PullSource for Audio {
    fn provide(&mut self, input: &str, max: usize) -> Result<InputData<'_>, PullError> {
        assert_eq!(input, "audio");
        self.asked += 1;
        let n = max.min(self.data.len() - self.at);
        let from = self.at;
        self.at += n;
        Ok(InputData::F32(&self.data[from..from + n]))
    }
}

fn audio(n: usize, stream: u64) -> Vec<f32> {
    let mut rng = Pcg32::new(SEED, 100 + stream);
    (0..n).map(|_| rng.normal() as f32 * 0.2).collect()
}

#[test]
fn pull_mode_gives_the_samples_of_push_mode_and_asks_the_source_once_per_round() {
    let data = audio(20_000, 4);
    let doc = pair_transmitter().to_string();
    // Push mode.
    let mut pushed = build(&doc);
    let mut want = Vec::new();
    let mut bytes = vec![0u8; 20_000];
    for part in data.chunks(1000) {
        pushed.push("audio", InputData::F32(part)).unwrap();
        loop {
            let n = pushed.read_raw("rf", &mut bytes).unwrap();
            if n == 0 {
                break;
            }
            want.extend_from_slice(&bytes[..n]);
        }
    }
    assert_eq!(want.len(), data.len() * 2);
    // Pull mode: the same bytes.
    let mut pulled = build(&doc);
    let mut source = Audio {
        data: data.clone(),
        at: 0,
        asked: 0,
    };
    let mut got = Vec::new();
    let mut buf = vec![0u8; 3000];
    loop {
        let (n, _) = pulled.pull("rf", &mut buf, &mut source).unwrap();
        if n == 0 {
            break;
        }
        got.extend_from_slice(&buf[..n]);
    }
    assert_eq!(got.len(), want.len());
    assert!(got == want, "pull mode differs from push mode");
    // The source of both paths is the one input: it is asked for data about once per request.
    assert!(
        source.asked <= got.len() / 100 + 10,
        "{} calls",
        source.asked
    );
}

#[test]
fn a_burst_flushes_and_resets_both_paths() {
    let data = audio(2_000, 5);
    let mut p = build(&pair_transmitter().to_string());
    let pull_burst = |p: &mut Pipeline, source: &mut Audio| {
        let mut bytes = Vec::new();
        let mut buf = vec![0u8; 5000];
        let mut ended = false;
        while !ended {
            let (n, meta) = p.pull("rf", &mut buf, source).unwrap();
            if n == 0 && source.at >= source.data.len() {
                break;
            }
            bytes.extend_from_slice(&buf[..n]);
            ended = meta.flags.contains(BlockFlags::BURST_END);
        }
        bytes
    };
    p.begin_burst("rf", None).unwrap();
    let mut source = Audio {
        data: data.clone(),
        at: 0,
        asked: 0,
    };
    // The data of the burst, then the end of the burst with the tail.
    let mut buf = vec![0u8; 5000];
    let mut first = Vec::new();
    loop {
        let (n, _) = p.pull("rf", &mut buf, &mut source).unwrap();
        if n == 0 {
            break;
        }
        first.extend_from_slice(&buf[..n]);
    }
    p.end_burst("rf").unwrap();
    first.extend(pull_burst(&mut p, &mut source));
    // The second burst starts from reset nodes, the filter on the side branch included.
    p.begin_burst("rf", None).unwrap();
    let mut source2 = Audio {
        data,
        at: 0,
        asked: 0,
    };
    let mut second = Vec::new();
    loop {
        let (n, _) = p.pull("rf", &mut buf, &mut source2).unwrap();
        if n == 0 {
            break;
        }
        second.extend_from_slice(&buf[..n]);
    }
    p.end_burst("rf").unwrap();
    second.extend(pull_burst(&mut p, &mut source2));
    assert!(first.len() > 100, "the burst has data");
    assert_eq!(second, first, "the second burst equals the first");
    let ends = std::iter::from_fn(|| p.poll_event())
        .filter(|e| matches!(e.kind, EventKind::BurstEnd))
        .count();
    assert_eq!(ends, 2);
}

#[test]
fn the_buffers_of_the_two_ports_stay_bounded_over_ten_million_samples() {
    // The debug assertion of the runner checks the difference of the fills at every step.
    let doc = two_paths(
        "f32",
        vec![
            (
                "lp",
                json!({ "kind": "biquad", "params": { "cutoff_hz": 2000, "response": "lowpass" } }),
            ),
            ("d", json!({ "kind": "delay", "params": { "samples": 3 } })),
        ],
        json!({ "kind": "add" }),
    );
    let report = validate(&doc);
    assert!(report.is_ok(), "{}", report.to_json());
    assert!(report.nodes.iter().any(|n| n.name == "sum__align"));
    let mut p = build(&doc.to_string());
    let mut rng = Pcg32::new(SEED, 200);
    let block: Vec<f32> = (0..4096).map(|_| rng.normal() as f32 * 0.1).collect();
    let mut sizes = ChunkPlan::new(SEED, 4096).sizes();
    let mut out = vec![0.0f32; 8192];
    let (mut pushed, mut read) = (0usize, 0usize);
    while pushed < 10_000_000 {
        let n = sizes.next().unwrap();
        p.push("x", InputData::F32(&block[..n])).unwrap();
        pushed += n;
        loop {
            let got = p.read_stream("out", &mut out).unwrap();
            if got == 0 {
                break;
            }
            read += got;
        }
    }
    assert_eq!(read, pushed, "nothing is held back at the end of the run");
}

#[test]
fn groups_keep_the_ports_and_a_cut_into_one_port_is_a_second_clock() {
    let mut doc = json!({
        "sdrk": 1, "name": "t", "settings": { "max_chunk": 1024 },
        "groups": { "front": {}, "back": {} },
        "inputs": { "x": { "type": "f32", "rate_hz": 48000 } },
        "nodes": {
            "lp": { "kind": "biquad", "group": "back",
                    "params": { "cutoff_hz": 1000, "response": "lowpass" } },
            "sum": { "kind": "add", "group": "back" },
            "hold": { "kind": "delay", "group": "front", "params": { "samples": 0 } }
        },
        "edges": [
            { "from": "x", "to": "sum", "port": "a" },
            { "from": "x", "to": "lp" },
            { "from": "lp", "to": "sum", "port": "b" },
            { "from": "sum", "to": "hold" },
            { "from": "hold", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    });
    // Both ports in one group: the part keeps the ports.
    let spec = PipelineSpec::from_json(&doc.to_string()).unwrap();
    let parts = split_by_groups(&spec, &Registry::standard()).unwrap();
    let back = &parts.iter().find(|(g, _)| g == "back").unwrap().1;
    assert!(
        back.edges
            .iter()
            .any(|e| e.port == Some(sdrk_pipeline::PortName::B))
    );
    for (group, part) in &parts {
        let report = Registry::standard().validate(part, &BuildOptions::default());
        assert!(report.is_ok(), "{group}: {}", report.to_json());
    }
    // The branch into port `b` in another group arrives through a second input of the part.
    doc["nodes"]["lp"]["group"] = json!("front");
    let spec = PipelineSpec::from_json(&doc.to_string()).unwrap();
    let report = split_by_groups(&spec, &Registry::standard()).unwrap_err();
    assert_eq!(report.errors[0].path, "groups.back");
    assert!(
        report.errors[0].message.contains("clock_domain"),
        "{}",
        report.errors[0].message
    );
}
