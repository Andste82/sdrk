//! The committed JSON Schema and the example flows.
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

use sdrk_pipeline::{BuildOptions, PipelineSpec, Registry, SCHEMA_FILE, SCHEMA_ID};
use serde_json::{Value, json};

use common::{flow, workspace_file};

const FLOWS: [&str; 9] = [
    "fm_receiver",
    "spectrum_only",
    "resample_audio",
    "fm_receiver_grouped",
    "fm_transmitter",
    "nfm_receiver",
    "am_receiver",
    "airband_receiver",
    "ssb_receiver",
];

fn schema() -> Value {
    Registry::standard().json_schema()
}

#[test]
fn the_committed_schema_is_up_to_date() {
    let committed = std::fs::read_to_string(workspace_file(SCHEMA_FILE)).unwrap();
    let fresh = Registry::standard().json_schema_text();
    assert!(
        committed.replace("\r\n", "\n") == fresh,
        "schema/sdrk-pipeline-v1.json is out of date: run `cargo xtask schema`"
    );
}

#[test]
fn the_schema_has_the_documented_shape() {
    let s = schema();
    assert_eq!(s["$schema"], "https://json-schema.org/draft/2020-12/schema");
    assert_eq!(s["$id"], SCHEMA_ID);
    assert_eq!(s["properties"]["sdrk"]["const"], 1);
    // One alternative per block kind, in the order of the registry.
    let kinds: Vec<&str> = s["properties"]["nodes"]["additionalProperties"]["oneOf"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k["properties"]["kind"]["const"].as_str().unwrap())
        .collect();
    let registry = Registry::standard();
    let expected: Vec<&str> = registry.descriptions().map(|d| d.kind).collect();
    assert_eq!(kinds, expected);
    // A parameter carries its limits, default and the documented description.
    let ddc = &s["properties"]["nodes"]["additionalProperties"]["oneOf"]
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["properties"]["kind"]["const"] == "ddc")
        .unwrap()["properties"]["params"];
    assert_eq!(ddc["additionalProperties"], false);
    assert_eq!(ddc["required"], json!(["decimation"]));
    let decimation = &ddc["properties"]["decimation"];
    assert_eq!(decimation["type"], "integer");
    assert_eq!(decimation["minimum"], 1);
    assert_eq!(decimation["maximum"], 1024);
    let text = decimation["description"].as_str().unwrap();
    assert!(
        text.contains("Typical:") && text.ends_with("Build-time."),
        "{text}"
    );
    let frequency = &ddc["properties"]["frequency_hz"];
    assert_eq!(frequency["default"], 0.0);
    assert!(
        frequency["description"]
            .as_str()
            .unwrap()
            .ends_with("Runtime.")
    );
    assert!(
        frequency["description"]
            .as_str()
            .unwrap()
            .contains("Unit: Hz.")
    );
}

#[test]
fn the_schema_keys_are_sorted() {
    fn check(v: &Value) {
        match v {
            Value::Object(m) => {
                let keys: Vec<&String> = m.keys().collect();
                let mut sorted = keys.clone();
                sorted.sort();
                assert_eq!(keys, sorted);
                m.values().for_each(check);
            }
            Value::Array(a) => a.iter().for_each(check),
            _ => {}
        }
    }
    check(&schema());
}

#[test]
fn every_example_flow_validates_against_the_schema() {
    let validator = jsonschema::validator_for(&schema()).unwrap();
    for name in FLOWS {
        let doc: Value = serde_json::from_str(&flow(name)).unwrap();
        let errors: Vec<String> = validator.iter_errors(&doc).map(|e| e.to_string()).collect();
        assert!(errors.is_empty(), "{name}: {errors:?}");
        // And the pipeline's own validation agrees.
        let report = Registry::standard().validate_json(&flow(name), &BuildOptions::default());
        assert!(report.is_ok(), "{name}: {}", report.to_json());
    }
}

#[test]
fn the_schema_rejects_mistakes_in_a_flow() {
    let validator = jsonschema::validator_for(&schema()).unwrap();
    let good: Value = serde_json::from_str(&flow("fm_receiver")).unwrap();
    assert!(validator.is_valid(&good));
    let mutate = |f: &dyn Fn(&mut Value)| {
        let mut doc = good.clone();
        f(&mut doc);
        doc
    };
    let bad = [
        mutate(&|d| d["sdrk"] = json!(2)),
        mutate(&|d| d["nodes"]["tune"]["params"]["decimation"] = json!(0)),
        mutate(&|d| d["nodes"]["tune"]["params"]["decimation"] = json!(10.5)),
        mutate(&|d| d["nodes"]["tune"]["params"]["decimaton"] = json!(10)),
        mutate(&|d| d["nodes"]["tune"]["kind"] = json!("dcc")),
        mutate(&|d| d["nodes"]["fft"]["params"]["window"] = json!("triangle")),
        mutate(&|d| d["nodes"]["tune"]["params"] = json!({})),
        mutate(&|d| d["settings"]["precision"] = json!("f64")),
        mutate(&|d| d["settings"]["max_chunk"] = json!(0)),
        mutate(&|d| d["inputs"]["rf"]["rate_hz"] = json!(0)),
        mutate(&|d| d["inputs"]["rf"]["format"] = json!("cu9")),
        mutate(&|d| d["outputs"]["audio"]["drop"] = json!("block")),
        mutate(&|d| d["extra"] = json!(1)),
        mutate(&|d| {
            let n = d["nodes"].as_object_mut().unwrap();
            let v = n.remove("tune").unwrap();
            n.insert("Tune".into(), v);
        }),
    ];
    for (i, doc) in bad.iter().enumerate() {
        assert!(!validator.is_valid(doc), "mutation {i} should be rejected");
    }
    // Accepted alternatives: a fraction as rate, a custom format, a choice by name.
    let ok = mutate(&|d| {
        d["inputs"]["rf"]["rate_hz"] = json!("4800000/2");
        d["inputs"]["rf"]["format"] = json!({
            "container": "i16", "bits": 12, "align": "lsb", "layout": "interleaved",
            "endian": "le", "full_scale": { "explicit": 2048 }
        });
        d["nodes"]["fft"]["params"]["window"] = json!("kaiser");
    });
    assert!(validator.is_valid(&ok));
    let report = Registry::standard().validate_json(&ok.to_string(), &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
}

#[test]
fn every_description_appears_in_the_schema_with_all_its_parameters() {
    let s = schema();
    let alternatives = s["properties"]["nodes"]["additionalProperties"]["oneOf"]
        .as_array()
        .unwrap();
    for desc in Registry::standard().descriptions() {
        let alt = alternatives
            .iter()
            .find(|a| a["properties"]["kind"]["const"] == desc.kind)
            .unwrap_or_else(|| panic!("kind {} missing", desc.kind));
        assert_eq!(alt["description"], desc.summary);
        let props = alt["properties"]["params"]["properties"]
            .as_object()
            .unwrap();
        assert_eq!(props.len(), desc.params.len(), "{}", desc.kind);
        for p in desc.params {
            assert!(props.contains_key(p.name), "{}.{}", desc.kind, p.name);
        }
    }
}

#[test]
fn a_spec_survives_the_schema_round_trip() {
    for name in FLOWS {
        let spec = PipelineSpec::from_json(&flow(name)).unwrap();
        let doc: Value = serde_json::from_str(&spec.to_json()).unwrap();
        let validator = jsonschema::validator_for(&schema()).unwrap();
        assert!(validator.is_valid(&doc), "{name}");
    }
}
