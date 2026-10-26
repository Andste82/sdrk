//! Validation: the invalid-flow corpus, all errors at once, limits, cost and memory figures.
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

use std::collections::BTreeSet;

use sdrk_pipeline::{BuildOptions, DiagCode, Limits, PipelineSpec, Registry, Report};
use serde_json::{Value, json};

use common::{build, flow};

fn validate(json: &str) -> Report {
    Registry::standard().validate_json(json, &BuildOptions::default())
}

fn validate_value(v: &Value) -> Report {
    validate(&v.to_string())
}

fn codes(report: &Report) -> BTreeSet<(String, String)> {
    report
        .errors
        .iter()
        .map(|d| (d.code.as_str().to_string(), d.path.clone()))
        .collect()
}

/// A small valid flow: c32 input, one gain node, one stream output.
fn base() -> Value {
    json!({
        "sdrk": 1, "name": "t",
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": { "g": { "kind": "scale", "params": { "gain_db": -6 } } },
        "edges": [ { "from": "iq", "to": "g" }, { "from": "g", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

#[test]
fn every_invalid_flow_of_the_corpus_is_rejected_with_the_expected_diagnostic() {
    let dir = common::workspace_file("crates/pipeline/sdrk-pipeline/tests/invalid");
    let mut seen_codes = BTreeSet::new();
    let mut files = 0;
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        if path.extension().is_none_or(|e| e != "json") || name.ends_with(".expected.json") {
            continue;
        }
        files += 1;
        let stem = name.trim_end_matches(".json");
        let text = std::fs::read_to_string(&path).unwrap();
        let expected: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join(format!("{stem}.expected.json"))).unwrap(),
        )
        .unwrap();
        let want: BTreeSet<(String, String)> = expected["errors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| {
                (
                    e["code"].as_str().unwrap().to_string(),
                    e["path"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        let report = validate(&text);
        assert_eq!(codes(&report), want, "{name}: {}", report.to_json());
        assert_eq!(
            report.errors.len(),
            want.len(),
            "{name}: no extra diagnostics"
        );
        for d in &report.errors {
            assert!(!d.message.is_empty(), "{name}: message");
            seen_codes.insert(d.code);
        }
        // `build` returns the same report as the error.
        let spec = serde_json::from_str::<PipelineSpec>(&text);
        if let Ok(spec) = spec {
            let err = Registry::standard()
                .build(&spec, &BuildOptions::default())
                .unwrap_err();
            assert_eq!(codes(&err), want, "{name}: build");
        }
    }
    assert_eq!(files, 21, "one file per diagnostic code");
    let all = [
        DiagCode::Syntax,
        DiagCode::SchemaVersion,
        DiagCode::UnknownField,
        DiagCode::InvalidName,
        DiagCode::DuplicateName,
        DiagCode::UnknownKind,
        DiagCode::UnknownParam,
        DiagCode::WrongType,
        DiagCode::OutOfRange,
        DiagCode::MissingParam,
        DiagCode::InvalidValue,
        DiagCode::UnknownEndpoint,
        DiagCode::MultipleInputs,
        DiagCode::MissingInput,
        DiagCode::DanglingOutput,
        DiagCode::Cycle,
        DiagCode::TypeMismatch,
        DiagCode::RateUnsupported,
        DiagCode::LimitExceeded,
        DiagCode::UnknownGroup,
        DiagCode::BackendUnavailable,
    ];
    for code in all {
        assert!(seen_codes.contains(&code), "no corpus file for {code:?}");
    }
}

#[test]
fn the_base_flow_is_valid() {
    let report = validate_value(&base());
    assert!(report.is_ok(), "{}", report.to_json());
    assert!(report.warnings.is_empty());
    assert_eq!(report.nodes.len(), 1);
}

#[test]
fn three_independent_mistakes_give_three_diagnostics() {
    let mut doc = base();
    doc["nodes"]["g"]["params"] = json!({ "gain_db": 100, "gane": 1 });
    doc["nodes"]["h"] = json!({ "kind": "scalee" });
    let report = validate_value(&doc);
    assert_eq!(report.errors.len(), 3, "{}", report.to_json());
    let got = codes(&report);
    assert!(got.contains(&("out_of_range".into(), "nodes.g.params.gain_db".into())));
    assert!(got.contains(&("unknown_param".into(), "nodes.g.params.gane".into())));
    assert!(got.contains(&("unknown_kind".into(), "nodes.h.kind".into())));
}

#[test]
fn graph_mistakes_are_reported_together_after_the_structure_is_sound() {
    let mut doc = base();
    doc["edges"].as_array_mut().unwrap().extend([
        json!({ "from": "g", "to": "ghost" }), // unknown endpoint
        json!({ "from": "iq", "to": "out" }),  // second incoming edge
    ]);
    doc["nodes"]["h"] = json!({ "kind": "scale" }); // not connected at all
    let report = validate_value(&doc);
    let got = codes(&report);
    assert_eq!(report.errors.len(), 4, "{}", report.to_json());
    assert!(got.contains(&("unknown_endpoint".into(), "edges[2].to".into())));
    assert!(got.contains(&("multiple_inputs".into(), "edges[3].to".into())));
    assert!(got.contains(&("missing_input".into(), "nodes.h".into())));
    assert!(got.contains(&("dangling_output".into(), "nodes.h".into())));
}

#[test]
fn a_structure_error_stops_the_graph_stage() {
    let mut doc = base();
    doc["nodes"]["g"]["kind"] = json!("nope");
    doc["edges"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "from": "g", "to": "ghost" }));
    let report = validate_value(&doc);
    assert_eq!(
        codes(&report),
        BTreeSet::from([("unknown_kind".into(), "nodes.g.kind".into())])
    );
}

#[test]
fn syntax_errors_carry_line_and_column() {
    let report = validate("{\n  \"sdrk\": 1,\n  \"name\": \n}");
    assert_eq!(report.errors.len(), 1);
    let d = &report.errors[0];
    assert_eq!(d.code, DiagCode::Syntax);
    assert!(
        d.message.contains("line 4") && d.message.contains("column"),
        "{}",
        d.message
    );
    // Not an object at all.
    assert_eq!(validate("[1, 2]").errors[0].code, DiagCode::WrongType);
    assert_eq!(validate("").errors[0].code, DiagCode::Syntax);
}

#[test]
fn structure_problems_get_precise_paths() {
    let cases: Vec<(Value, &str, DiagCode)> = vec![
        (
            json!({ "sdrk": 1, "name": "x", "settings": { "max_chunk": 0 } }),
            "settings.max_chunk",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "settings": { "precision": "f64" } }),
            "settings.precision",
            DiagCode::InvalidValue,
        ),
        (json!({ "sdrk": 1, "name": 5 }), "name", DiagCode::WrongType),
        (json!({ "sdrk": 1 }), "name", DiagCode::MissingParam),
        (json!({ "name": "x" }), "sdrk", DiagCode::SchemaVersion),
        (
            json!({ "sdrk": "1", "name": "x" }),
            "sdrk",
            DiagCode::SchemaVersion,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "inputs": { "a": { "type": "c32" } } }),
            "inputs.a.rate_hz",
            DiagCode::MissingParam,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "inputs": { "a": { "type": "c32", "rate_hz": 0 } } }),
            "inputs.a.rate_hz",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "inputs": { "a": { "type": "c32", "rate_hz": "x/2" } } }),
            "inputs.a.rate_hz",
            DiagCode::InvalidValue,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "inputs": { "a": { "type": "c32", "rate_hz": true } } }),
            "inputs.a.rate_hz",
            DiagCode::WrongType,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "inputs": { "a": { "format": "cu9", "rate_hz": 5 } } }),
            "inputs.a.format",
            DiagCode::InvalidValue,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "inputs": { "a": { "rate_hz": 5 } } }),
            "inputs.a",
            DiagCode::InvalidValue,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "inputs": { "a": { "type": "c32", "format": "cu8", "rate_hz": 5 } } }),
            "inputs.a",
            DiagCode::InvalidValue,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "inputs": { "a": { "type": "c64", "rate_hz": 5 } } }),
            "inputs.a.type",
            DiagCode::InvalidValue,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "nodes": { "a": {} } }),
            "nodes.a.kind",
            DiagCode::MissingParam,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "nodes": { "a": { "kind": "scale", "params": 3 } } }),
            "nodes.a.params",
            DiagCode::WrongType,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "edges": [ { "from": "a" } ] }),
            "edges[0].to",
            DiagCode::MissingParam,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "edges": [ 7 ] }),
            "edges[0]",
            DiagCode::WrongType,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "outputs": { "o": {} } }),
            "outputs.o.type",
            DiagCode::MissingParam,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "outputs": { "o": { "type": "stream", "drop": "x" } } }),
            "outputs.o.drop",
            DiagCode::InvalidValue,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "outputs": { "o": { "type": "stream", "capacity": -1 } } }),
            "outputs.o.capacity",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "outputs": { "o": { "type": "stream", "capacity": 1.5 } } }),
            "outputs.o.capacity",
            DiagCode::InvalidValue,
        ),
        (
            json!({ "sdrk": 1, "name": "x", "description": "x".repeat(2001) }),
            "description",
            DiagCode::OutOfRange,
        ),
    ];
    for (doc, path, code) in cases {
        let report = validate_value(&doc);
        assert!(
            report
                .errors
                .iter()
                .any(|d| d.path == path && d.code == code),
            "{doc}: expected {code:?} at {path}, got {}",
            report.to_json()
        );
    }
}

#[test]
fn typed_validation_catches_what_the_types_cannot_express() {
    let registry = Registry::standard();
    let opts = BuildOptions::default();
    let mut doc = base();
    doc["outputs"]["out"] = json!({ "type": "stream", "queue": 3 });
    let report = validate_value(&doc);
    assert_eq!(
        codes(&report),
        BTreeSet::from([("invalid_value".into(), "outputs.out.queue".into())])
    );
    doc["outputs"]["out"] = json!({ "type": "frames", "capacity": 3 });
    assert!(
        codes(&validate_value(&doc))
            .contains(&("invalid_value".into(), "outputs.out.capacity".into()))
    );
    doc["outputs"]["out"] = json!({ "type": "stream", "capacity": 0 });
    assert!(
        codes(&validate_value(&doc))
            .contains(&("out_of_range".into(), "outputs.out.capacity".into()))
    );
    // A spec built in Rust is checked the same way.
    let mut spec: PipelineSpec = serde_json::from_value(base()).unwrap();
    spec.sdrk = 7;
    spec.settings.max_chunk = 0;
    let report = registry.validate(&spec, &opts);
    assert_eq!(report.errors.len(), 2);
}

#[test]
fn names_are_unique_across_inputs_nodes_outputs_and_automatic_nodes() {
    let doc = json!({
        "sdrk": 1, "name": "t",
        "inputs": { "rf": { "format": "cu8", "rate_hz": 48000 } },
        "nodes": { "rf__convert": { "kind": "scale" } },
        "edges": [ { "from": "rf", "to": "rf__convert" }, { "from": "rf__convert", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    });
    let report = validate_value(&doc);
    assert!(
        codes(&report).contains(&("duplicate_name".into(), "nodes.rf__convert".into())),
        "{}",
        report.to_json()
    );
    let doc = json!({
        "sdrk": 1, "name": "t",
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": { "q": { "kind": "scale" } },
        "edges": [ { "from": "iq", "to": "q" }, { "from": "q", "to": "q" } ],
        "outputs": { "iq": { "type": "stream" } }
    });
    assert!(codes(&validate_value(&doc)).contains(&("duplicate_name".into(), "outputs.iq".into())));
}

#[test]
fn types_are_checked_at_every_edge() {
    // frames cannot feed a node
    let doc = json!({
        "sdrk": 1, "name": "t",
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": { "f": { "kind": "spectrum" }, "g": { "kind": "scale" } },
        "edges": [ { "from": "iq", "to": "f" }, { "from": "f", "to": "g" }, { "from": "g", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    });
    assert!(codes(&validate_value(&doc)).contains(&("type_mismatch".into(), "edges[1].to".into())));
    // a stream output cannot take frames, and the other way round
    let doc = json!({
        "sdrk": 1, "name": "t",
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": { "f": { "kind": "spectrum" }, "g": { "kind": "scale" } },
        "edges": [ { "from": "iq", "to": "f" }, { "from": "f", "to": "a" }, { "from": "iq", "to": "g" }, { "from": "g", "to": "b" } ],
        "outputs": { "a": { "type": "stream" }, "b": { "type": "frames" } }
    });
    let got = codes(&validate_value(&doc));
    assert!(
        got.contains(&("type_mismatch".into(), "outputs.a.type".into())),
        "{got:?}"
    );
    assert!(
        got.contains(&("type_mismatch".into(), "outputs.b.type".into())),
        "{got:?}"
    );
    // the input type must fit the precision
    let doc = json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "q15" },
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 } },
        "nodes": { "g": { "kind": "scale" } },
        "edges": [ { "from": "iq", "to": "g" }, { "from": "g", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    });
    assert!(
        codes(&validate_value(&doc)).contains(&("type_mismatch".into(), "inputs.iq.type".into()))
    );
    // an fm_mod needs a real signal; a resampler takes both
    let doc = json!({
        "sdrk": 1, "name": "t",
        "inputs": { "iq": { "type": "c32", "rate_hz": 48000 }, "au": { "type": "f32", "rate_hz": 48000 } },
        "nodes": { "m": { "kind": "fm_mod" }, "r": { "kind": "resampler", "params": { "rate_out_hz": 96000 } },
                   "r2": { "kind": "resampler", "params": { "rate_out_hz": 96000 } } },
        "edges": [ { "from": "iq", "to": "m" }, { "from": "m", "to": "o1" }, { "from": "au", "to": "r" },
                   { "from": "r", "to": "o2" }, { "from": "iq", "to": "r2" }, { "from": "r2", "to": "o3" } ],
        "outputs": { "o1": { "type": "stream" }, "o2": { "type": "stream" }, "o3": { "type": "stream" } }
    });
    let report = validate_value(&doc);
    assert_eq!(
        codes(&report),
        BTreeSet::from([("type_mismatch".into(), "edges[0].to".into())]),
        "{}",
        report.to_json()
    );
}

#[test]
fn rates_propagate_exactly() {
    let report = validate(&flow("fm_receiver"));
    assert!(report.is_ok(), "{}", report.to_json());
    let by_name = |n: &str| report.nodes.iter().find(|r| r.name == n).unwrap();
    assert_eq!(by_name("rf__convert").rate_in.num(), 2_400_000);
    assert_eq!(by_name("tune").rate_out.num(), 240_000);
    assert_eq!(by_name("demod").rate_in.num(), 240_000);
    assert_eq!(by_name("demod").rate_out.num(), 48_000);
    assert!(by_name("rf__convert").auto);
    assert!(!by_name("tune").auto);
    // a fractional input rate stays exact
    let doc = json!({
        "sdrk": 1, "name": "t",
        "inputs": { "iq": { "type": "c32", "rate_hz": "1000000/3" } },
        "nodes": { "d": { "kind": "ddc", "params": { "decimation": 4 } } },
        "edges": [ { "from": "iq", "to": "d" }, { "from": "d", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    });
    let report = validate_value(&doc);
    assert!(report.is_ok(), "{}", report.to_json());
    let d = &report.nodes[0];
    assert_eq!((d.rate_out.num(), d.rate_out.den()), (250_000, 3));
}

#[test]
fn unused_inputs_are_warnings_not_errors() {
    let mut doc = base();
    doc["inputs"]["spare"] = json!({ "type": "c32", "rate_hz": 8000 });
    let report = validate_value(&doc);
    assert!(report.is_ok());
    assert_eq!(report.warnings.len(), 1);
    assert_eq!(report.warnings[0].path, "inputs.spare");
    assert_eq!(report.warnings[0].code, DiagCode::DanglingOutput);
}

#[test]
fn automatic_conversions_appear_in_the_report_not_in_the_config() {
    let doc = json!({
        "sdrk": 1, "name": "t",
        "inputs": { "rf": { "format": "cs16", "rate_hz": 48000 } },
        "nodes": { "g": { "kind": "scale" } },
        "edges": [ { "from": "rf", "to": "g" }, { "from": "g", "to": "out" } ],
        "outputs": { "out": { "type": "stream", "format": "cs8" } }
    });
    let report = validate_value(&doc);
    assert!(report.is_ok(), "{}", report.to_json());
    let names: Vec<(&str, &str, bool)> = report
        .nodes
        .iter()
        .map(|n| (n.name.as_str(), n.kind.as_str(), n.auto))
        .collect();
    assert_eq!(
        names,
        vec![
            ("rf__convert", "convert", true),
            ("g", "scale", false),
            ("out__convert_out", "convert_out", true)
        ]
    );
    let pipeline = build(&doc.to_string());
    let stats: Vec<&str> = pipeline.stats().iter().map(|s| s.name.as_str()).collect();
    assert_eq!(stats, vec!["rf__convert", "g", "out__convert_out"]);
    let cfg = pipeline.config();
    assert_eq!(cfg.nodes.len(), 1);
    assert_eq!(
        cfg.to_json(),
        PipelineSpec::from_json(&doc.to_string()).unwrap().to_json()
    );
}

#[test]
fn raw_formats_that_cannot_stream_are_rejected_at_the_input() {
    for format in ["rf32", "rs16", "cs16_planar"] {
        let doc = json!({
            "sdrk": 1, "name": "t",
            "inputs": { "rf": { "format": format, "rate_hz": 48000 } },
            "nodes": { "g": { "kind": "scale" } },
            "edges": [ { "from": "rf", "to": "g" }, { "from": "g", "to": "out" } ],
            "outputs": { "out": { "type": "stream" } }
        });
        let report = validate_value(&doc);
        assert!(
            codes(&report).contains(&("invalid_value".into(), "inputs.rf.format".into())),
            "{format}: {}",
            report.to_json()
        );
    }
    // A real stream cannot get a raw output format either.
    let doc = json!({
        "sdrk": 1, "name": "t",
        "inputs": { "au": { "type": "f32", "rate_hz": 48000 } },
        "nodes": { "r": { "kind": "resampler", "params": { "rate_out_hz": 96000 } } },
        "edges": [ { "from": "au", "to": "r" }, { "from": "r", "to": "out" } ],
        "outputs": { "out": { "type": "stream", "format": "rs16" } }
    });
    assert!(
        codes(&validate_value(&doc))
            .contains(&("invalid_value".into(), "outputs.out.format".into()))
    );
}

#[test]
fn the_report_is_machine_readable_json() {
    let mut doc = base();
    doc["nodes"]["g"]["params"]["gain_db"] = json!(100);
    let report = validate_value(&doc);
    let v: Value = serde_json::from_str(&report.to_json()).unwrap();
    assert_eq!(v["errors"][0]["code"], "out_of_range");
    assert_eq!(v["errors"][0]["path"], "nodes.g.params.gain_db");
    assert!(
        v["errors"][0]["message"]
            .as_str()
            .unwrap()
            .contains("gain_db")
    );
    assert!(v["warnings"].is_array());
    assert!(v["memory_bytes"].is_number());
    let ok = validate(&flow("fm_receiver"));
    let v: Value = serde_json::from_str(&ok.to_json()).unwrap();
    assert_eq!(v["nodes"][0]["name"], "rf__convert");
    assert_eq!(v["nodes"][0]["auto"], true);
    assert_eq!(v["nodes"][0]["rate_in"], 2_400_000);
    assert!(v["nodes"][1]["delay_samples_in"].is_string());
    // every code serializes as snake_case
    for e in [
        DiagCode::Syntax,
        DiagCode::SchemaVersion,
        DiagCode::LimitExceeded,
    ] {
        assert_eq!(serde_json::to_value(e).unwrap(), e.as_str());
    }
}

#[test]
fn the_estimate_of_validate_matches_the_built_report() {
    let registry = Registry::standard();
    for name in ["fm_receiver", "spectrum_only", "resample_audio"] {
        let text = flow(name);
        let estimate = registry.validate_json(&text, &BuildOptions::default());
        assert!(estimate.is_ok());
        let pipeline = registry
            .build_json(&text, &BuildOptions::default())
            .unwrap();
        let built = pipeline.report();
        assert_eq!(built.nodes.len(), estimate.nodes.len(), "{name}");
        for (a, b) in estimate.nodes.iter().zip(&built.nodes) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.rate_out, b.rate_out, "{name}.{}", a.name);
            assert_eq!(
                a.delay_samples_in.reduced(),
                b.delay_samples_in.reduced(),
                "{name}.{}: delay of validate and of the built block",
                a.name
            );
        }
        assert!(built.memory_bytes > 0);
        assert!(
            built.memory_bytes.abs_diff(estimate.memory_bytes) <= estimate.memory_bytes / 20,
            "{name}: estimate {} vs built {}",
            estimate.memory_bytes,
            built.memory_bytes
        );
        assert_eq!(built.cost_per_input_sample, estimate.cost_per_input_sample);
    }
}

/// Estimate and built delay agree for every kind and both precisions.
#[test]
fn delays_of_validate_match_the_blocks_for_every_kind() {
    let kinds = [
        ("scale", json!({})),
        ("agc", json!({})),
        (
            "am_demod",
            json!({ "audio_decimation": 2, "audio_cutoff_hz": 5000 }),
        ),
        ("ssb_demod", json!({ "audio_decimation": 2 })),
        ("dc_block", json!({ "cutoff_hz": 50 })),
        ("fir", json!({ "cutoff_hz": 4000, "transition_hz": 1000 })),
        ("ddc", json!({ "decimation": 4, "frequency_hz": 1000 })),
        ("duc", json!({ "interpolation": 3, "frequency_hz": 100 })),
        (
            "fm_demod",
            json!({ "audio_decimation": 2, "audio_cutoff_hz": 5000 }),
        ),
        (
            "resampler",
            json!({ "rate_out_hz": 44100, "max_taps": 8192 }),
        ),
        (
            "resampler",
            json!({ "rate_out_hz": 44100, "tunable": true, "ppm": 100 }),
        ),
        ("resampler", json!({ "rate_out_hz": 96000 })),
    ];
    let registry = Registry::standard();
    for precision in ["f32", "q15"] {
        for (kind, params) in &kinds {
            let ty = if *kind == "fm_mod" { "f32" } else { "c32" };
            let ty = match (ty, precision) {
                ("c32", "q15") => "cq15",
                ("f32", "q15") => "q15",
                (t, _) => t,
            };
            let doc = json!({
                "sdrk": 1, "name": "t", "settings": { "precision": precision, "max_chunk": 1024 },
                "inputs": { "in": { "type": ty, "rate_hz": 48000 } },
                "nodes": { "n": { "kind": kind, "params": params } },
                "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
                "outputs": { "out": { "type": "stream" } }
            });
            let text = doc.to_string();
            let estimate = registry.validate_json(&text, &BuildOptions::default());
            assert!(
                estimate.is_ok(),
                "{kind} {precision}: {}",
                estimate.to_json()
            );
            let pipeline = registry
                .build_json(&text, &BuildOptions::default())
                .unwrap();
            let (a, b) = (&estimate.nodes[0], &pipeline.report().nodes[0]);
            assert_eq!(
                a.delay_samples_in.reduced(),
                b.delay_samples_in.reduced(),
                "{kind} {precision}"
            );
            assert_eq!(a.rate_out, b.rate_out, "{kind} {precision}");
        }
    }
}

// ------------------------------------------------------------------ limits

fn fm_spec() -> PipelineSpec {
    PipelineSpec::from_json(&flow("fm_receiver")).unwrap()
}

fn with_limits(limits: Limits) -> BuildOptions {
    BuildOptions { limits }
}

#[test]
fn every_limit_triggers_limit_exceeded() {
    let registry = Registry::standard();
    let spec = fm_spec();
    let ok = registry.validate(&spec, &BuildOptions::default());
    assert!(ok.is_ok());
    let cases = [
        (
            "max_nodes",
            Limits {
                max_nodes: 3,
                ..Limits::default()
            },
        ),
        (
            "max_memory_bytes",
            Limits {
                max_memory_bytes: 1000,
                ..Limits::default()
            },
        ),
        (
            "max_cost_per_input_sample",
            Limits {
                max_cost_per_input_sample: 1.0,
                ..Limits::default()
            },
        ),
        (
            "max_taps_per_node",
            Limits {
                max_taps_per_node: 10,
                ..Limits::default()
            },
        ),
    ];
    for (name, limits) in cases {
        let report = registry.validate(&spec, &with_limits(limits));
        assert!(
            report
                .errors
                .iter()
                .any(|d| d.code == DiagCode::LimitExceeded),
            "{name}: {}",
            report.to_json()
        );
        assert!(
            registry.build(&spec, &with_limits(limits)).is_err(),
            "{name}"
        );
    }
    // The taps limit names the node.
    let report = registry.validate(
        &spec,
        &with_limits(Limits {
            max_taps_per_node: 300,
            ..Limits::default()
        }),
    );
    let paths: Vec<&str> = report.errors.iter().map(|d| d.path.as_str()).collect();
    assert_eq!(paths, vec!["nodes.tune"], "{}", report.to_json());
    // Exactly at the limit is fine.
    let nodes = ok.nodes.len();
    let exact = registry.validate(
        &spec,
        &with_limits(Limits {
            max_nodes: nodes,
            ..Limits::default()
        }),
    );
    assert!(exact.is_ok());
    let exact = registry.validate(
        &spec,
        &with_limits(Limits {
            max_memory_bytes: ok.memory_bytes,
            ..Limits::default()
        }),
    );
    assert!(exact.is_ok());
}

#[test]
fn memory_and_cost_are_positive_and_grow_with_chunk_and_taps() {
    let registry = Registry::standard();
    let mut last_memory = 0;
    for max_chunk in [64usize, 256, 1024, 4096, 16384, 65536] {
        let mut spec = fm_spec();
        spec.settings.max_chunk = max_chunk;
        let report = registry.validate(&spec, &BuildOptions::default());
        assert!(report.is_ok(), "{}", report.to_json());
        assert!(report.memory_bytes > 0 && report.cost_per_input_sample > 0.0);
        assert!(
            report.memory_bytes > last_memory,
            "memory grows with max_chunk"
        );
        last_memory = report.memory_bytes;
        // The built pipeline agrees about the order of magnitude.
        if max_chunk <= 4096 {
            let p = registry.build(&spec, &BuildOptions::default()).unwrap();
            assert!(p.report().memory_bytes >= p_min(&spec));
        }
    }
    // A narrower filter needs more taps: more memory and more cost.
    let mut last = (0usize, 0.0f64);
    for transition in [40_000.0, 20_000.0, 10_000.0, 5_000.0] {
        let mut spec = fm_spec();
        let node = spec.nodes.get_mut("tune").unwrap();
        node.params
            .insert("transition_hz".into(), json!(transition));
        node.params.insert("max_taps".into(), json!(20_000));
        let report = registry.validate(&spec, &BuildOptions::default());
        assert!(report.is_ok(), "{}", report.to_json());
        assert!(report.memory_bytes > last.0, "memory grows with the taps");
        assert!(
            report.cost_per_input_sample > last.1,
            "cost grows with the taps"
        );
        last = (report.memory_bytes, report.cost_per_input_sample);
    }
}

/// A lower bound for the memory: the input buffer alone.
fn p_min(spec: &PipelineSpec) -> usize {
    spec.settings.max_chunk * 2
}

#[test]
fn the_cost_scales_with_the_rate_at_the_node() {
    // A fir behind a decimating ddc runs at the lower rate and costs proportionally less.
    let doc = |before: bool| {
        let (a, b) = if before { ("f", "d") } else { ("d", "f") };
        json!({
            "sdrk": 1, "name": "t",
            "inputs": { "iq": { "type": "c32", "rate_hz": 2400000 } },
            "nodes": {
                "d": { "kind": "ddc", "params": { "decimation": 10 } },
                "f": { "kind": "fir", "params": { "cutoff_hz": 50000, "transition_hz": 20000 } }
            },
            "edges": [ { "from": "iq", "to": a }, { "from": a, "to": b }, { "from": b, "to": "out" } ],
            "outputs": { "out": { "type": "stream" } }
        })
    };
    let fir_first = validate_value(&doc(true));
    let ddc_first = validate_value(&doc(false));
    assert!(fir_first.is_ok() && ddc_first.is_ok());
    assert!(fir_first.cost_per_input_sample > 2.5 * ddc_first.cost_per_input_sample);
}

#[test]
fn building_an_invalid_description_returns_the_report() {
    let registry = Registry::standard();
    let mut doc = base();
    doc["nodes"]["g"]["params"]["gain_db"] = json!(100);
    let spec: PipelineSpec = serde_json::from_value(doc.clone()).unwrap();
    let report = registry.build(&spec, &BuildOptions::default()).unwrap_err();
    assert_eq!(report.errors[0].code, DiagCode::OutOfRange);
    match registry.build_json(&doc.to_string(), &BuildOptions::default()) {
        Err(sdrk_pipeline::Error::Build(r)) => {
            assert_eq!(r.errors[0].path, "nodes.g.params.gain_db");
        }
        other => panic!("{other:?}"),
    }
}
