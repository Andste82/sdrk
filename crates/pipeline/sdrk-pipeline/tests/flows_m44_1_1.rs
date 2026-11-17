//! M44.1.1 test 5 (seed 862): the example flows that named `order` before M44.1.1
//! (`qpsk_transmitter`, `qpsk_receiver`, `qam16_transmitter`, `qam16_receiver`) name
//! `constellation`, load and validate as JSON and as YAML (the YAML twin is the same
//! specification), and their modulators give the output of M44.1.
//!
//! The modulator of each transmitter flow (the `psk_mod` node of `qpsk_transmitter`, and the
//! `qam_mod` part of the `qam_frame_encode` node of `qam16_transmitter`), read from the JSON and
//! from the YAML flow and run as a one-node flow in pieces of 1 to 1000 soft bits (seed 862),
//! gives the recorded output of M44.1 on the soft bits of its case (`tests/baseline/m44_1`,
//! tables `psk_mod` and `qam_mod`, the cases at 8 samples per symbol, roll-off 0.35 and span 10:
//! the seed 841 and the streams of `qam_baseline_m44_1`). The whole `qam16` pair keeps its
//! baseline in `qam_loop` (the transmitter's bytes and the receiver's soft bits); the receiver of
//! `qpsk_receiver`, `psk_demod`, is not changed by M44.1.1 beyond the name of `constellation`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::large_stack_frames
)]

mod common;

use sdrk::C32;
use sdrk_pipeline::{BuildOptions, InputData, PipelineSpec, Registry};
use sdrk_testkit::Pcg32;
use sdrk_testkit::baseline::{self, Baseline, Digest};
use serde_json::{Map, Value, json};

use common::{flow, workspace_file};

/// The seed of the pieces.
const SEED: u64 = 862;
/// The seed of the modulator inputs of the M44.1 baselines (`qam_baseline_m44_1`).
const SEED_MOD: u64 = 841;
/// Symbols per modulator case of the M44.1 baselines.
const MOD_SYMBOLS: usize = 700;

/// The flows, the node that names the constellation and its value.
const FLOWS: [(&str, &str, &str); 4] = [
    ("qpsk_transmitter", "mod", "qpsk"),
    ("qpsk_receiver", "demod", "qpsk"),
    ("qam16_transmitter", "tx", "qam16"),
    ("qam16_receiver", "rx", "qam16"),
];

fn text(name: &str, ext: &str) -> String {
    std::fs::read_to_string(workspace_file(&format!("examples/flows/{name}.{ext}"))).unwrap()
}

/// The specification of a flow as JSON, from its `json` or its `yaml` file.
fn spec(name: &str, ext: &str) -> Value {
    if ext == "json" {
        serde_json::from_str(&flow(name)).unwrap()
    } else {
        yaml_spec(name)
    }
}

#[cfg(feature = "yaml")]
fn yaml_spec(name: &str) -> Value {
    serde_json::from_str(
        &PipelineSpec::from_yaml(&text(name, "yaml"))
            .unwrap()
            .to_json(),
    )
    .unwrap()
}

#[cfg(not(feature = "yaml"))]
fn yaml_spec(name: &str) -> Value {
    spec(name, "json")
}

/// The file kinds of the flows: JSON, and YAML with the feature `yaml`.
fn exts() -> &'static [&'static str] {
    if cfg!(feature = "yaml") {
        &["json", "yaml"]
    } else {
        &["json"]
    }
}

#[test]
fn the_flows_name_constellation_and_validate_as_json_and_yaml() {
    let registry = Registry::standard();
    for (name, node, value) in FLOWS {
        for ext in exts() {
            let doc = spec(name, ext);
            let report = registry.validate_json(&doc.to_string(), &BuildOptions::default());
            assert!(report.is_ok(), "{name}.{ext}: {}", report.to_json());
            let params = doc["nodes"][node]["params"].as_object().unwrap();
            assert_eq!(params["constellation"], json!(value), "{name}.{ext}");
            assert!(
                !doc.to_string().contains("\"order\""),
                "{name}.{ext} still names order"
            );
            assert!(
                registry
                    .build_json(&doc.to_string(), &BuildOptions::default())
                    .is_ok()
            );
        }
        // The JSON file is the canonical form of the specification.
        let json_spec = PipelineSpec::from_json(&text(name, "json")).unwrap();
        assert_eq!(json_spec.to_json() + "\n", text(name, "json"), "{name}");
        assert_eq!(
            spec(name, "yaml"),
            spec(name, "json"),
            "{name}: the YAML twin"
        );
    }
}

/// Random soft bits `+-1` of seed 841 and `stream`, with an extreme value of the sign rule at
/// every 37th position: the inputs of the modulator cases of `qam_baseline_m44_1`.
fn mod_bits(n: usize, stream: u64) -> Vec<f32> {
    const EXTREMES: [f32; 10] = [
        0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-30,
        0.5,
        -2.0,
    ];
    let mut rng = Pcg32::new(SEED_MOD, stream);
    (0..n)
        .map(|i| {
            let r = rng.next_u32();
            if i % 37 == 36 {
                EXTREMES[(r % 10) as usize]
            } else if r & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Runs a one-node flow of `kind` with `params` on `bits`, pushed in pieces of 1 to 1000 soft
/// bits, and returns the digest of its output.
fn run_node(kind: &str, params: &Map<String, Value>, bits: &[f32], stream: u64) -> Digest {
    let doc = json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": "f32", "rate_hz": 48_000 } },
        "nodes": { "n": { "kind": kind, "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream", "capacity": 65536 } }
    });
    let mut p = Registry::standard()
        .build_json(&doc.to_string(), &BuildOptions::default())
        .unwrap();
    let mut rng = Pcg32::new(SEED, stream);
    let mut out = Vec::new();
    let mut buf = vec![C32::default(); 65536];
    let mut rest = bits;
    while !rest.is_empty() {
        let n = (1 + rng.next_u32() as usize % 1000).min(rest.len());
        p.push("in", InputData::F32(&rest[..n])).unwrap();
        rest = &rest[n..];
        loop {
            let k = p.read_stream::<C32>("out", &mut buf).unwrap();
            if k == 0 {
                break;
            }
            out.extend_from_slice(&buf[..k]);
        }
    }
    Digest::of_c32(&out)
}

#[test]
fn the_modulators_of_the_flows_give_the_output_of_m44_1() {
    let level = baseline::level(sdrk_backend_simd::level_name());
    let psk = Baseline::load("m44_1", "psk_mod", level).unwrap();
    let qam = Baseline::load("m44_1", "qam_mod", level).unwrap();
    for (i, ext) in exts().iter().enumerate() {
        let stream = 10 * i as u64;
        // `qpsk_transmitter`: coherent QPSK, the case of stream 1027 (the fourth of QPSK).
        let doc = spec("qpsk_transmitter", ext);
        let node = &doc["nodes"]["mod"];
        assert_eq!(node["kind"], json!("psk_mod"));
        let params = node["params"].as_object().unwrap();
        let bits = mod_bits(MOD_SYMBOLS * 2, 1027);
        let d = run_node("psk_mod", params, &bits, stream);
        psk.check("psk4_coh_plain_sps8_r0.35_s10", &d)
            .unwrap_or_else(|e| panic!("qpsk_transmitter.{ext}: {e}"));
        // `qam16_transmitter`: the modulator parameters of `qam_frame_encode`, the case of
        // stream 3.
        let doc = spec("qam16_transmitter", ext);
        let node = &doc["nodes"]["tx"];
        assert_eq!(node["kind"], json!("qam_frame_encode"));
        let all = node["params"].as_object().unwrap();
        let params: Map<String, Value> = ["constellation", "samples_per_symbol", "rolloff", "span"]
            .iter()
            .map(|k| ((*k).to_string(), all[*k].clone()))
            .collect();
        let bits = mod_bits(MOD_SYMBOLS * 4, 3);
        let d = run_node("qam_mod", &params, &bits, stream + 1);
        qam.check("qam16_sps8_r0.35_s10", &d)
            .unwrap_or_else(|e| panic!("qam16_transmitter.{ext}: {e}"));
    }
}
