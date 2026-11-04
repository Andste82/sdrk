//! Fuzz target: `from_json`, `validate_json` and, if the description is valid, `build` and a
//! push of a few bytes. None of them may panic or hang.
//!
//! The first bytes of the input also choose a small flow with a node of two inputs (the kind,
//! the ports of its edges, `align`, a filter on one path and the rate of the second input), so
//! that the port model, the checks of rates, clocks and delays and the automatic alignment are
//! reached without the mutator having to find the JSON syntax first.
#![no_main]

use libfuzzer_sys::fuzz_target;
use sdrk_pipeline::{BuildOptions, InputData, PipelineSpec, Registry};

/// A flow with one node of two inputs, shaped by the first bytes of `data`.
fn two_input_flow(data: &[u8]) -> String {
    let byte = |i: usize| data.get(i).copied().unwrap_or(0);
    let kind = ["add", "mul", "make_complex", "stereo_pll", "scale"][usize::from(byte(0)) % 5];
    let ty = ["f32", "c32"][usize::from(byte(1)) % 2];
    let port = |b: u8| match b % 5 {
        0 => String::new(),
        1 => r#", "port": "a""#.to_string(),
        2 => r#", "port": "b""#.to_string(),
        3 => r#", "port": "c""#.to_string(),
        _ => r#", "port": 3"#.to_string(),
    };
    let align = match byte(4) % 3 {
        0 => "",
        1 => r#", "align": "none""#,
        _ => r#", "align": "auto""#,
    };
    let path = match byte(5) % 5 {
        0 => r#"{ "from": "x", "to": "n"#.to_string() + "\" " + &port(byte(3)) + " }",
        1 => r#"{ "from": "x", "to": "d" }, { "from": "d", "to": "n"#.to_string() + "\" " + &port(byte(3)) + " }",
        2 => r#"{ "from": "x", "to": "f" }, { "from": "f", "to": "n"#.to_string() + "\" " + &port(byte(3)) + " }",
        3 => r#"{ "from": "y", "to": "n"#.to_string() + "\" " + &port(byte(3)) + " }",
        _ => r#"{ "from": "n", "to": "n"#.to_string() + "\" " + &port(byte(3)) + " }",
    };
    let rate_y = [48_000u32, 24_000, 192_000][usize::from(byte(6)) % 3];
    // Only the helper node that the second path uses is declared.
    let helper = match byte(5) % 5 {
        1 => format!(
            r#", "d": {{ "kind": "delay", "params": {{ "samples": {} }} }}"#,
            u32::from(byte(7)) * 300
        ),
        2 => format!(
            r#", "f": {{ "kind": "decimate", "params": {{ "factor": {} }} }}"#,
            1 + u32::from(byte(8)) % 5
        ),
        _ => String::new(),
    };
    format!(
        r#"{{ "sdrk": 1, "name": "f",
  "inputs": {{ "x": {{ "type": "{ty}", "rate_hz": 192000 }}, "y": {{ "type": "{ty}", "rate_hz": {rate_y} }} }},
  "nodes": {{ "n": {{ "kind": "{kind}"{align} }}{helper} }},
  "edges": [
    {{ "from": "x", "to": "n"{} }},
    {path},
    {{ "from": "n", "to": "out" }}
  ],
  "outputs": {{ "out": {{ "type": "stream" }} }} }}"#,
        port(byte(2)),
    )
}

fn exercise(text: &str) {
    let registry = Registry::standard();
    let mut opts = BuildOptions::default();
    opts.limits.max_memory_bytes = 64 * 1024 * 1024;
    let report = registry.validate_json(text, &opts);
    let _ = report.to_json();
    if report.is_ok() {
        if let Ok(mut pipeline) = registry.build_json(text, &opts) {
            let names: Vec<String> = pipeline.config().inputs.keys().cloned().collect();
            for name in names {
                let _ = pipeline.push(&name, InputData::C32(&[sdrk::C32::new(0.5, -0.5); 64]));
                let _ = pipeline.push(&name, InputData::F32(&[0.25; 64]));
            }
        }
    }
}

fuzz_target!(|data: &[u8]| {
    exercise(&two_input_flow(data));
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let registry = Registry::standard();
    // Keep a build of a huge but valid description from exhausting the fuzzer's memory.
    let mut opts = BuildOptions::default();
    opts.limits.max_memory_bytes = 64 * 1024 * 1024;
    let _ = PipelineSpec::from_json(text);
    let report = registry.validate_json(text, &opts);
    let _ = report.to_json();
    if report.is_ok() {
        if let Ok(mut pipeline) = registry.build_json(text, &opts) {
            let names: Vec<String> = pipeline.config().inputs.keys().cloned().collect();
            for name in names {
                let _ = pipeline.push(&name, InputData::Raw(&[0, 1, 2, 3, 4, 5, 6, 7]));
                let _ = pipeline.push(&name, InputData::C32(&[sdrk::C32::new(0.5, -0.5); 8]));
            }
        }
    }
});
