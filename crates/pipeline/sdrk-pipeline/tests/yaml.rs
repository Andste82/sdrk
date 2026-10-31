//! YAML input: the same model as JSON, with comments, and line and column in errors.
#![cfg(feature = "yaml")]
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
    clippy::doc_markdown
)]

mod common;

use std::collections::BTreeSet;

use sdrk_pipeline::{BuildOptions, DiagCode, Error, InputData, PipelineSpec, Registry};

use common::{cu8_recording, workspace_file};

const FLOWS: [&str; 10] = [
    "fm_receiver",
    "spectrum_only",
    "resample_audio",
    "fm_receiver_grouped",
    "fm_transmitter",
    "nfm_receiver",
    "am_receiver",
    "airband_receiver",
    "ssb_receiver",
    "real_input_fm_receiver",
];

fn text(name: &str, ext: &str) -> String {
    std::fs::read_to_string(workspace_file(&format!("examples/flows/{name}.{ext}"))).unwrap()
}

#[test]
fn every_example_flow_has_a_yaml_twin_with_comments() {
    for name in FLOWS {
        let yaml = text(name, "yaml");
        assert!(
            yaml.lines()
                .filter(|l| l.trim_start().starts_with('#'))
                .count()
                >= 2,
            "{name}.yaml has comments"
        );
        let from_yaml = PipelineSpec::from_yaml(&yaml).unwrap();
        let from_json = PipelineSpec::from_json(&text(name, "json")).unwrap();
        assert_eq!(from_yaml, from_json, "{name}");
        // The canonical JSON of the YAML spec is the JSON example.
        assert_eq!(from_yaml.to_json() + "\n", text(name, "json"), "{name}");
    }
}

#[test]
fn a_yaml_flow_validates_and_builds_like_the_json_flow() {
    let registry = Registry::standard();
    for name in FLOWS {
        let report = registry.validate_yaml(&text(name, "yaml"), &BuildOptions::default());
        assert!(report.is_ok(), "{name}: {}", report.to_json());
        let other = registry.validate_json(&text(name, "json"), &BuildOptions::default());
        assert_eq!(report, other, "{name}");
    }
    let mut p = registry
        .build_yaml(&text("fm_receiver", "yaml"), &BuildOptions::default())
        .unwrap();
    p.push("rf", InputData::Raw(&cu8_recording(4000))).unwrap();
    assert!(p.available("audio").unwrap() > 0);
    assert!(matches!(
        registry.build_yaml("sdrk: 2\nname: x\n", &BuildOptions::default()),
        Err(Error::Build(_))
    ));
}

#[test]
fn a_syntax_error_reports_line_and_column() {
    let err = PipelineSpec::from_yaml("sdrk: 1\nname: ok\nsettings: [\n  oops\n").unwrap_err();
    let Error::Yaml(message) = &err else {
        panic!("{err:?}")
    };
    assert!(
        message.contains("line ") && message.contains("column "),
        "{message}"
    );
    let err = PipelineSpec::from_yaml("sdrk: 1\nname: x\nbogus: 3\n").unwrap_err();
    let Error::Yaml(message) = &err else {
        panic!("{err:?}")
    };
    assert!(
        message.contains("bogus") && message.contains("line 3"),
        "{message}"
    );
    // As a diagnostic with the position.
    let report = Registry::standard().validate_yaml("a: b: c\n", &BuildOptions::default());
    assert_eq!(report.errors.len(), 1);
    assert_eq!(report.errors[0].code, DiagCode::Syntax);
    assert!(
        report.errors[0].message.contains("line 1"),
        "{}",
        report.errors[0].message
    );
    assert!(report.errors[0].message.contains("column "));
}

#[test]
fn the_diagnostics_of_yaml_and_json_are_identical() {
    // JSON is YAML: the files of the invalid corpus go through the YAML parser too.
    let dir = workspace_file("crates/pipeline/sdrk-pipeline/tests/invalid");
    let registry = Registry::standard();
    let mut compared = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        let json = path.extension().is_some_and(|e| e == "json");
        if !json || name.ends_with(".expected.json") || name == "syntax.json" {
            continue;
        }
        let body = std::fs::read_to_string(&path).unwrap();
        let from_json = registry.validate_json(&body, &BuildOptions::default());
        let from_yaml = registry.validate_yaml(&body, &BuildOptions::default());
        let key = |r: &sdrk_pipeline::Report| {
            r.errors
                .iter()
                .map(|d| (d.code, d.path.clone()))
                .collect::<BTreeSet<_>>()
        };
        assert_eq!(key(&from_yaml), key(&from_json), "{name}");
        compared += 1;
    }
    assert!(compared >= 19, "{compared}");
}

#[test]
fn yaml_anchors_aliases_and_flow_style_work() {
    let yaml = "
# anchors are plain YAML
sdrk: 1
name: aliases
inputs: { iq: { type: c32, rate_hz: '96000/2' } }   # an exact fraction
nodes:
  g: &gain
    kind: scale
    params: { gain_db: -6.5 }
  h: *gain
edges: [ { from: iq, to: g }, { from: g, to: h }, { from: h, to: out } ]
outputs: { out: { type: stream } }
";
    let spec = PipelineSpec::from_yaml(yaml).unwrap();
    assert_eq!(spec.inputs["iq"].rate_hz.0.num(), 48000);
    assert_eq!(spec.nodes["g"], spec.nodes["h"]);
    assert!(
        Registry::standard()
            .validate(&spec, &BuildOptions::default())
            .is_ok()
    );
}
