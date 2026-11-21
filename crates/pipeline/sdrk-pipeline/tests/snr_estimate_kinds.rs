//! The kind `snr_estimate` (M44.4 §1): registration without kernels, the cost 10 per symbol, the
//! rate and the delay, the type check (`c32` only), `invalid_value` for `bpsk` and `8psk` and
//! `out_of_range` for the window, the build parameters, and a push-mode flow whose output is its
//! input and whose reading `snr_db` is that of the block, in pieces of up to 1000 symbols
//! (seed 902).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::needless_pass_by_value,
    clippy::large_stack_frames
)]

mod common;

use sdrk::{C32, Delay, Rate};
use sdrk_blocks::{Block, SnrEstimateBlock, SnrEstimateConfig, SymbolMapConstellation};
use sdrk_core::kernels::constellation::Qam;
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry, VecStorage};
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::build;

/// The seed of the symbols, the noise and the pieces.
const SEED: u64 = 902;

fn single(ty: &str, params: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": 10000 } },
        "nodes": { "snr": { "kind": "snr_estimate", "params": params } },
        "edges": [ { "from": "in", "to": "snr" }, { "from": "snr", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

fn validate(doc: &Value) -> sdrk_pipeline::Report {
    Registry::standard().validate_json(&doc.to_string(), &BuildOptions::default())
}

#[test]
fn the_kind_is_registered_without_kernels_and_costs_ten() {
    let registry = Registry::standard();
    let f = registry.factory("snr_estimate").unwrap();
    assert_eq!(f.kernels().len(), 0);
    assert_eq!(f.description().kind, "snr_estimate");
    assert_eq!(registry.descriptions().count(), 84);
    let report = validate(&single("c32", json!({ "constellation": "qam256" })));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.cost_per_input_sample, 10.0);
    let node = &report.nodes[0];
    assert_eq!(node.rate_out, Rate::hz(10_000).unwrap());
    assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO);
}

#[test]
fn the_kind_checks_its_input_and_its_parameters() {
    let report = validate(&single("cq15", json!({})));
    assert_eq!(
        report.errors[0].code,
        DiagCode::TypeMismatch,
        "{}",
        report.to_json()
    );
    for (params, code) in [
        (json!({ "constellation": "bpsk" }), DiagCode::InvalidValue),
        (json!({ "constellation": "8psk" }), DiagCode::InvalidValue),
        (json!({ "window": 63 }), DiagCode::OutOfRange),
        (json!({ "window": 1_048_577 }), DiagCode::OutOfRange),
    ] {
        let report = validate(&single("c32", params.clone()));
        let e = report.errors.first().expect("an error");
        assert_eq!(e.code, code, "{params}");
        assert!(
            e.path.starts_with("nodes.snr.params."),
            "{params}: {}",
            e.path
        );
    }
    for c in ["qpsk", "qam16", "qam64", "qam256", "qam1024"] {
        let report = validate(&single("c32", json!({ "constellation": c, "window": 64 })));
        assert!(report.is_ok(), "{c}: {}", report.to_json());
    }
}

#[test]
fn a_flow_passes_the_symbols_and_reports_the_estimate_of_the_block() {
    let q = Qam::Qam64;
    let mut rng = Pcg32::new(SEED, 10);
    let x: Vec<C32> = (0..9 * 1024 + 300)
        .map(|_| {
            let p = q.point(rng.below(64));
            let sd = 0.05;
            C32::new(
                p.re + (rng.normal() * sd) as f32,
                p.im + (rng.normal() * sd) as f32,
            )
        })
        .collect();
    let cfg = SnrEstimateConfig {
        constellation: SymbolMapConstellation::Qam64,
        window: 1024,
    };
    let mut est = SnrEstimateBlock::new(&cfg, Rate::hz(10_000).unwrap(), &mut VecStorage).unwrap();
    let mut want = vec![C32::default(); x.len()];
    est.process(&x, &mut want).unwrap();
    let doc = single("c32", json!({ "constellation": "qam64", "window": 1024 }));
    let mut pipe = build(&doc.to_string());
    let st = &pipe.stats()[0];
    assert_eq!(st.name, "snr");
    assert_eq!(st.readings.len(), 1);
    assert_eq!(st.readings[0].0, "snr_db");
    assert!(st.readings[0].1.is_nan());
    let mut got = Vec::new();
    let mut buf = vec![C32::default(); 4096];
    let mut sizes = Pcg32::new(SEED, 11);
    let mut pos = 0;
    while pos < x.len() {
        let n = (1 + sizes.below(1000) as usize).min(x.len() - pos);
        pipe.push("in", InputData::C32(&x[pos..pos + n])).unwrap();
        pos += n;
        let k = pipe.read_stream::<C32>("out", &mut buf).unwrap();
        got.extend_from_slice(&buf[..k]);
    }
    assert_eq!(got, x, "the output is the input");
    let st = &pipe.stats()[0];
    assert_eq!(st.readings[0].1.to_bits(), est.snr_db().to_bits());
    // 64-QAM with noise of 0.005 per symbol: 23 dB.
    assert!((est.snr_db() - 23.0).abs() < 0.3, "{}", est.snr_db());
    // Both parameters are build parameters.
    assert!(pipe.set_param("snr", "window", json!(2048)).is_err());
    assert!(
        pipe.set_param("snr", "constellation", json!("qam16"))
            .is_err()
    );
    pipe.reset();
    assert!(pipe.stats()[0].readings[0].1.is_nan());
}
