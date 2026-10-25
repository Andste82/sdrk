//! The data model: JSON round trip, names, rates, formats.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::assert_is_empty,
    clippy::unreadable_literal,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::redundant_closure_for_method_calls
)]

mod common;

use sdrk_device::formats;
use sdrk_pipeline::{
    DropPolicy, Error, FORMAT_NAMES, FormatName, FormatSpec, PipelineSpec, Precision, RateHz,
    is_valid_name,
};
use serde_json::json;

use common::flow;

const FLOWS: [&str; 8] = [
    "fm_receiver",
    "spectrum_only",
    "resample_audio",
    "fm_receiver_grouped",
    "fm_transmitter",
    "nfm_receiver",
    "am_receiver",
    "airband_receiver",
];

#[test]
fn every_example_flow_round_trips() {
    for name in FLOWS {
        let text = flow(name);
        let spec = PipelineSpec::from_json(&text).unwrap();
        let again = PipelineSpec::from_json(&spec.to_json()).unwrap();
        assert_eq!(spec, again, "{name}: parse(to_json(x)) == x");
        // The example files are canonical: the bytes are identical.
        assert_eq!(spec.to_json() + "\n", text, "{name}: canonical bytes");
    }
}

#[test]
fn the_canonical_form_orders_keys_and_sorts_maps() {
    let json = r#"{
        "outputs": { "z_out": { "type": "stream" }, "a_out": { "type": "stream" } },
        "edges": [],
        "nodes": { "zz": { "kind": "scale", "params": { "gain_db": 1, "gain_db2": 2 } }, "aa": { "kind": "scale" } },
        "inputs": { "b": { "rate_hz": 10, "type": "c32" }, "a": { "type": "f32", "rate_hz": "30/4" } },
        "settings": { "max_chunk": 16 },
        "name": "x",
        "sdrk": 1
    }"#;
    let spec = PipelineSpec::from_json(json).unwrap();
    let out = spec.to_json();
    let position = |needle: &str| {
        out.find(needle)
            .unwrap_or_else(|| panic!("{needle} in {out}"))
    };
    // Field order: sdrk, name, settings, inputs, nodes, edges, outputs.
    assert!(position("\"sdrk\"") < position("\"name\""));
    assert!(position("\"name\"") < position("\"settings\""));
    assert!(position("\"settings\"") < position("\"inputs\""));
    assert!(position("\"inputs\"") < position("\"nodes\""));
    assert!(position("\"nodes\"") < position("\"edges\""));
    assert!(position("\"edges\"") < position("\"outputs\""));
    // Maps sorted by key.
    assert!(position("\"a\"") < position("\"b\""));
    assert!(position("\"aa\"") < position("\"zz\""));
    assert!(position("\"a_out\"") < position("\"z_out\""));
    assert!(position("\"gain_db\"") < position("\"gain_db2\""));
    // Defaults are written out; the fraction is reduced on load.
    assert!(out.contains("\"precision\": \"f32\""));
    assert!(out.contains("\"15/2\""));
}

#[test]
fn unknown_fields_are_errors() {
    for json in [
        r#"{"sdrk":1,"name":"x","extra":1}"#,
        r#"{"sdrk":1,"name":"x","settings":{"precision":"f32","chunk":4}}"#,
        r#"{"sdrk":1,"name":"x","nodes":{"a":{"kind":"scale","param":{}}}}"#,
        r#"{"sdrk":1,"name":"x","edges":[{"from":"a","to":"b","via":"c"}]}"#,
        r#"{"sdrk":1,"name":"x","outputs":{"o":{"type":"stream","size":3}}}"#,
        r#"{"sdrk":1,"name":"x","inputs":{"i":{"type":"c32","rate_hz":1,"unit":"hz"}}}"#,
    ] {
        assert!(
            matches!(PipelineSpec::from_json(json), Err(Error::Json(_))),
            "{json}"
        );
    }
}

#[test]
fn rates_are_exact_and_reduced() {
    let parse = |s: &str| serde_json::from_str::<RateHz>(s);
    assert_eq!(
        parse("2400000").unwrap().0,
        sdrk::Rate::hz(2_400_000).unwrap()
    );
    assert_eq!(
        parse("\"96000/2\"").unwrap().0,
        sdrk::Rate::hz(48_000).unwrap()
    );
    let third = parse("\"1000/3\"").unwrap();
    assert_eq!((third.0.num(), third.0.den()), (1000, 3));
    assert_eq!(serde_json::to_string(&third).unwrap(), "\"1000/3\"");
    assert_eq!(
        serde_json::to_string(&parse("\"6/3\"").unwrap()).unwrap(),
        "2"
    );
    for bad in [
        "0",
        "-5",
        "1.5",
        "\"abc\"",
        "\"1/0\"",
        "\"0/5\"",
        "\"1/2/3\"",
        "\"\"",
        "true",
        "null",
    ] {
        assert!(parse(bad).is_err(), "{bad}");
    }
}

#[test]
fn names_follow_the_pattern() {
    for ok in ["a", "rf", "fft_1", "a0_b", &"x".repeat(64)] {
        assert!(is_valid_name(ok), "{ok}");
    }
    for bad in [
        "",
        "1a",
        "_a",
        "A",
        "a-b",
        "a b",
        "ä",
        &"x".repeat(65),
        "a.b",
    ] {
        assert!(!is_valid_name(bad), "{bad}");
    }
}

#[test]
fn format_names_map_to_the_device_constants() {
    let pairs = [
        ("cu8", formats::CU8),
        ("cs8", formats::CS8),
        ("cs16", formats::CS16),
        ("cs12_lsb16", formats::CS12_LSB16),
        ("sc16_q11", formats::SC16_Q11),
        ("cs16_planar", formats::CS16_PLANAR),
        ("packed12", formats::PACKED12),
        ("packed10", formats::PACKED10),
        ("i2s32_msb", formats::I2S32_MSB),
        ("cf32", formats::CF32),
        ("rf32", formats::RF32),
        ("rs16", formats::RS16),
    ];
    assert_eq!(pairs.len(), FORMAT_NAMES.len());
    for (name, format) in pairs {
        assert_eq!(
            FormatName::from_name(name).unwrap().format(),
            format,
            "{name}"
        );
        let spec: FormatSpec = serde_json::from_value(json!(name)).unwrap();
        assert_eq!(spec.to_format().unwrap(), format);
        assert_eq!(serde_json::to_value(spec).unwrap(), json!(name));
    }
    assert!(FormatName::from_name("cu9").is_none());
}

#[test]
fn custom_formats_map_to_sample_formats() {
    let spec: FormatSpec = serde_json::from_value(json!({
        "container": "i16", "bits": 12, "align": "lsb", "layout": "interleaved",
        "endian": "le", "full_scale": { "explicit": 2048 }
    }))
    .unwrap();
    assert_eq!(spec.to_format().unwrap(), formats::SC16_Q11);
    let plain: FormatSpec = serde_json::from_value(json!({
        "container": "u8", "bits": 8, "align": "lsb", "layout": "interleaved", "endian": "le"
    }))
    .unwrap();
    assert_eq!(plain.to_format().unwrap(), formats::CU8);
    let swapped: FormatSpec = serde_json::from_value(json!({
        "container": "i16", "bits": 16, "align": "lsb", "layout": "interleaved",
        "endian": "be", "swap_iq": true
    }))
    .unwrap();
    let f = swapped.to_format().unwrap();
    assert!(f.swap_iq());
    // Not a possible format: more bits than the container.
    let bad: FormatSpec = serde_json::from_value(json!({
        "container": "i8", "bits": 12, "align": "lsb", "layout": "interleaved", "endian": "le"
    }))
    .unwrap();
    assert!(bad.to_format().is_err());
    // The custom form survives a round trip.
    assert_eq!(
        serde_json::from_value::<FormatSpec>(serde_json::to_value(spec).unwrap()).unwrap(),
        spec
    );
}

#[test]
fn settings_and_output_defaults() {
    let spec = PipelineSpec::from_json(r#"{"sdrk":1,"name":"x","outputs":{"o":{"type":"stream"},"f":{"type":"frames","queue":2}}}"#).unwrap();
    assert_eq!(spec.settings.precision, Precision::F32);
    assert_eq!(spec.settings.max_chunk, 4096);
    let o = &spec.outputs["o"];
    assert_eq!(o.drop_policy(), DropPolicy::Error);
    assert_eq!(o.capacity_samples(spec.settings.max_chunk), 16_384);
    assert_eq!(spec.outputs["f"].queue_frames(), 2);
    assert_eq!(spec.outputs["o"].queue_frames(), 4);
}

#[test]
fn descriptions_are_preserved() {
    let json = r#"{"sdrk":1,"name":"x","description":"top","inputs":{"i":{"type":"c32","rate_hz":5,"description":"in"}},"nodes":{"n":{"kind":"scale","description":"node"}},"outputs":{"o":{"type":"stream","description":"out"}}}"#;
    let spec = PipelineSpec::from_json(json).unwrap();
    let back = PipelineSpec::from_json(&spec.to_json()).unwrap();
    assert_eq!(back.description.as_deref(), Some("top"));
    assert_eq!(back.inputs["i"].description.as_deref(), Some("in"));
    assert_eq!(back.nodes["n"].description.as_deref(), Some("node"));
    assert_eq!(back.outputs["o"].description.as_deref(), Some("out"));
}
