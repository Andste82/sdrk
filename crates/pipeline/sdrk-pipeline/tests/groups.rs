//! Groups, backend preferences and `split_by_groups`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::assert_is_empty,
    clippy::unreadable_literal,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::redundant_closure_for_method_calls
)]

mod common;

use sdrk_pipeline::{
    BuildOptions, DiagCode, InputData, InputType, PipelineSpec, Registry, Report, split_by_groups,
};
use serde_json::{Value, json};

use common::{bits, build, cu8_recording, flow};

fn validate(v: &Value) -> Report {
    Registry::standard().validate_json(&v.to_string(), &BuildOptions::default())
}

fn grouped() -> Value {
    serde_json::from_str(&flow("fm_receiver_grouped")).unwrap()
}

#[test]
fn the_grouped_example_validates_and_lists_its_group_edges() {
    let report = validate(&grouped());
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.group_edges.len(), 1);
    let e = &report.group_edges[0];
    assert_eq!(
        (
            e.from.as_str(),
            e.to.as_str(),
            e.from_group.as_str(),
            e.to_group.as_str()
        ),
        ("tune", "demod", "front", "audio")
    );
    // The report is part of the JSON.
    let json: Value = serde_json::from_str(&report.to_json()).unwrap();
    assert_eq!(json["group_edges"][0]["from"], "tune");
    // Without groups there is nothing to list, and the key is absent.
    let plain = Registry::standard().validate_json(&flow("fm_receiver"), &BuildOptions::default());
    assert!(plain.group_edges.is_empty());
    let json: Value = serde_json::from_str(&plain.to_json()).unwrap();
    assert!(json.get("group_edges").is_none());
}

#[test]
fn nodes_without_a_group_are_in_the_implicit_main_group() {
    let mut v = grouped();
    v["nodes"]["demod"].as_object_mut().unwrap().remove("group");
    let report = validate(&v);
    assert!(report.is_ok());
    let main: Vec<_> = report
        .group_edges
        .iter()
        .map(|e| (e.from_group.as_str(), e.to_group.as_str()))
        .collect();
    assert_eq!(main, [("front", "main")]);
    // `main` may also be named explicitly without declaring it.
    v["nodes"]["demod"]["group"] = json!("main");
    assert!(validate(&v).is_ok());
}

#[test]
fn an_undeclared_group_is_an_error() {
    let mut v = grouped();
    v["nodes"]["tune"]["group"] = json!("back");
    let report = validate(&v);
    assert_eq!(report.errors.len(), 1, "{}", report.to_json());
    assert_eq!(report.errors[0].code, DiagCode::UnknownGroup);
    assert_eq!(report.errors[0].path, "nodes.tune.group");
    // Bad group names and unknown keys.
    let mut v = grouped();
    v["groups"]["Bad Name"] = json!({});
    assert!(
        validate(&v)
            .errors
            .iter()
            .any(|d| d.code == DiagCode::InvalidName && d.path == "groups.Bad Name")
    );
    let mut v = grouped();
    v["groups"]["front"]["cores"] = json!(2);
    assert!(
        validate(&v)
            .errors
            .iter()
            .any(|d| d.code == DiagCode::UnknownField && d.path == "groups.front.cores")
    );
}

#[test]
fn a_required_backend_that_is_not_available_fails() {
    let mut v = grouped();
    v["nodes"]["tune"]["backend"] = json!({ "require": "simd" });
    let report = validate(&v);
    assert_eq!(report.errors.len(), 1, "{}", report.to_json());
    assert_eq!(report.errors[0].code, DiagCode::BackendUnavailable);
    assert_eq!(report.errors[0].path, "nodes.tune.backend.require");
    assert!(
        report.errors[0].message.contains("fir_decimate"),
        "{}",
        report.errors[0].message
    );
    // The fallback is there for every kind of node of the example flows.
    for name in [
        "fm_receiver",
        "spectrum_only",
        "resample_audio",
        "fm_transmitter",
    ] {
        let mut v: Value = serde_json::from_str(&flow(name)).unwrap();
        for node in v["nodes"].as_object_mut().unwrap().values_mut() {
            node["backend"] = json!({ "require": "fallback" });
        }
        let report = validate(&v);
        assert!(report.is_ok(), "{name}: {}", report.to_json());
        assert!(report.warnings.is_empty(), "{name}: {}", report.to_json());
    }
    // The block lists the kernels it calls.
    let registry = Registry::standard();
    assert_eq!(
        registry.factory("ddc").unwrap().kernels(),
        ["rotate", "fir_decimate", "ddc"]
    );
}

#[test]
fn a_preferred_backend_that_is_not_available_is_a_warning() {
    let mut v = grouped();
    v["nodes"]["tune"]["backend"] = json!({ "prefer": "simd" });
    let report = validate(&v);
    assert!(report.is_ok());
    assert_eq!(report.warnings.len(), 1);
    assert_eq!(report.warnings[0].code, DiagCode::BackendUnavailable);
    assert_eq!(report.warnings[0].path, "nodes.tune.backend.prefer");
    // The pipeline still builds and runs: groups and preferences have no runtime effect.
    let mut p = build(&v.to_string());
    p.push("rf", InputData::Raw(&cu8_recording(4000))).unwrap();
    // A backend needs exactly one of the keys.
    v["nodes"]["tune"]["backend"] = json!({ "prefer": "fallback", "require": "fallback" });
    assert!(
        validate(&v)
            .errors
            .iter()
            .any(|d| d.path == "nodes.tune.backend")
    );
    v["nodes"]["tune"]["backend"] = json!({});
    assert!(!validate(&v).is_ok());
}

#[test]
fn groups_do_not_change_the_result() {
    let bytes = cu8_recording(40_000);
    let mut plain = build(&flow("fm_receiver"));
    let mut grouped = build(&flow("fm_receiver_grouped"));
    let run = |p: &mut sdrk_pipeline::Pipeline| {
        let mut audio = Vec::new();
        let mut buf = vec![0f32; 4096];
        for chunk in bytes.chunks(6000) {
            p.push("rf", InputData::Raw(chunk)).unwrap();
            let n = p.read_stream("audio", &mut buf).unwrap();
            audio.extend_from_slice(&buf[..n]);
        }
        audio
    };
    assert_eq!(bits(&run(&mut plain)), bits(&run(&mut grouped)));
}

#[test]
fn the_config_keeps_groups_and_backends() {
    let mut v = grouped();
    v["nodes"]["tune"]["backend"] = json!({ "prefer": "fallback" });
    let spec = PipelineSpec::from_json(&v.to_string()).unwrap();
    let again = PipelineSpec::from_json(&spec.to_json()).unwrap();
    assert_eq!(spec, again);
    assert_eq!(again.nodes["tune"].group.as_deref(), Some("front"));
    assert_eq!(again.groups.len(), 2);
    let p = Registry::standard()
        .build(&spec, &BuildOptions::default())
        .unwrap();
    assert_eq!(p.config(), spec);
}

#[test]
fn split_by_groups_makes_one_valid_description_per_group() {
    let registry = Registry::standard();
    let spec = PipelineSpec::from_json(&flow("fm_receiver_grouped")).unwrap();
    let parts = split_by_groups(&spec, &registry).unwrap();
    let names: Vec<&str> = parts.iter().map(|(g, _)| g.as_str()).collect();
    assert_eq!(names, ["audio", "front"]);
    for (group, part) in &parts {
        let report = registry.validate(part, &BuildOptions::default());
        assert!(report.is_ok(), "{group}: {}", report.to_json());
        assert!(part.groups.is_empty());
        assert!(part.nodes.values().all(|n| n.group.is_none()));
    }
    let audio = &parts[0].1;
    let front = &parts[1].1;
    // The producer group gets the output, the consumer group the input with the exact rate.
    assert!(front.outputs.contains_key("edge__tune__demod"));
    assert_eq!(front.nodes.keys().collect::<Vec<_>>(), ["tune"]);
    let edge_in = &audio.inputs["edge__tune__demod"];
    assert_eq!(edge_in.sample_type, Some(InputType::C32));
    assert_eq!(edge_in.rate_hz.0.num(), 240_000);
    assert_eq!(edge_in.rate_hz.0.den(), 1);
    // The input `rf` is copied into both groups; the outputs follow their source node.
    assert!(front.inputs.contains_key("rf") && audio.inputs.contains_key("rf"));
    assert!(audio.outputs.contains_key("audio") && audio.outputs.contains_key("spectrum"));
    assert_eq!(audio.nodes.keys().collect::<Vec<_>>(), ["demod", "fft"]);
    assert_eq!(parts[0].1.name, "fm-receiver-grouped.audio");
    // Both parts build.
    for (_, part) in &parts {
        registry.build(part, &BuildOptions::default()).unwrap();
    }
}

#[test]
fn the_parts_chained_by_hand_give_the_result_of_the_whole() {
    let registry = Registry::standard();
    let spec = PipelineSpec::from_json(&flow("fm_receiver_grouped")).unwrap();
    let parts = split_by_groups(&spec, &registry).unwrap();
    let mut audio = registry
        .build(&parts[0].1, &BuildOptions::default())
        .unwrap();
    let mut front = registry
        .build(&parts[1].1, &BuildOptions::default())
        .unwrap();
    let mut whole = build(&flow("fm_receiver_grouped"));
    let bytes = cu8_recording(40_000);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let mut buf = vec![0f32; 4096];
    let mut edge = vec![sdrk::C32::default(); 4096];
    for chunk in bytes.chunks(6000) {
        whole.push("rf", InputData::Raw(chunk)).unwrap();
        let n = whole.read_stream("audio", &mut buf).unwrap();
        a.extend_from_slice(&buf[..n]);
        front.push("rf", InputData::Raw(chunk)).unwrap();
        audio.push("rf", InputData::Raw(chunk)).unwrap();
        let n = front.read_stream("edge__tune__demod", &mut edge).unwrap();
        audio
            .push("edge__tune__demod", InputData::C32(&edge[..n]))
            .unwrap();
        let n = audio.read_stream("audio", &mut buf).unwrap();
        b.extend_from_slice(&buf[..n]);
    }
    assert!(!a.is_empty());
    assert_eq!(bits(&a), bits(&b));
}

#[test]
fn splitting_a_flow_without_groups_gives_the_main_group() {
    let registry = Registry::standard();
    let spec = PipelineSpec::from_json(&flow("fm_receiver")).unwrap();
    let parts = split_by_groups(&spec, &registry).unwrap();
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0].0, "main");
    assert_eq!(parts[0].1.nodes.len(), 3);
    assert!(
        registry
            .validate(&parts[0].1, &BuildOptions::default())
            .is_ok()
    );
}

#[test]
fn a_flow_with_input_to_output_edge_and_a_third_group_splits() {
    // Three groups in a row; an output fed directly by an input stays in `main`.
    let v = json!({
        "sdrk": 1, "name": "chain",
        "groups": { "a": {}, "b": {}, "c": {} },
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": {
            "one": { "kind": "scale", "group": "a" },
            "two": { "kind": "scale", "group": "b" },
            "three": { "kind": "scale", "group": "c" }
        },
        "edges": [
            { "from": "iq", "to": "one" }, { "from": "one", "to": "two" },
            { "from": "two", "to": "three" }, { "from": "three", "to": "out" },
            { "from": "iq", "to": "raw_out" }
        ],
        "outputs": { "out": { "type": "stream" }, "raw_out": { "type": "stream" } }
    });
    let registry = Registry::standard();
    let spec = PipelineSpec::from_json(&v.to_string()).unwrap();
    let parts = split_by_groups(&spec, &registry).unwrap();
    let names: Vec<&str> = parts.iter().map(|(g, _)| g.as_str()).collect();
    assert_eq!(names, ["a", "b", "c", "main"]);
    let report = registry.validate(&spec, &BuildOptions::default());
    assert_eq!(report.group_edges.len(), 2);
    assert!(parts[3].1.outputs.contains_key("raw_out"));
    assert!(parts[3].1.inputs.contains_key("iq"));
    assert!(parts[1].1.inputs.contains_key("edge__one__two"));
    assert!(parts[1].1.outputs.contains_key("edge__two__three"));
}

#[test]
fn a_flow_that_cannot_be_split_is_reported() {
    // A frames output cannot cross a group boundary: only streams can. A spectrum node cannot
    // feed a node anyway, so the only failing case is an invalid description.
    let registry = Registry::standard();
    let mut v = grouped();
    v["nodes"]["tune"]["params"]["decimation"] = json!(0);
    let spec: PipelineSpec = serde_json::from_value(v).unwrap();
    let report = split_by_groups(&spec, &registry).unwrap_err();
    assert!(!report.is_ok());
}
