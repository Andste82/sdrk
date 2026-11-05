//! The kinds `am_mod` and `ssb_mod` (M35): registration, costs, delays, type checks, runtime
//! parameters, and a push run that equals the block.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::needless_pass_by_value,
    clippy::large_stack_frames
)]

mod common;

use sdrk::{C32, CQ15, Delay, Q15, RealSample};
use sdrk_blocks::{
    AmModBlock, AmModConfig, Block, BlockSample, SsbModBlock, SsbModConfig, SsbSideband,
};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry};
use sdrk_testkit::{Pcg32, VecStorage};
use serde_json::{Value, json};

use common::build;

/// A one-node flow on an input of type `ty` at `rate` Hz.
fn single(kind: &str, params: Value, ty: &str, precision: &str, rate: u64) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": precision, "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": rate } },
        "nodes": { "n": { "kind": kind, "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

fn validate(doc: &Value) -> sdrk_pipeline::Report {
    Registry::standard().validate_json(&doc.to_string(), &BuildOptions::default())
}

#[test]
fn the_kinds_are_registered_with_their_kernels() {
    let registry = Registry::standard();
    let am = registry.factory("am_mod").unwrap();
    assert_eq!(am.kernels(), ["am_mod"]);
    assert_eq!(am.description().kind, "am_mod");
    let ssb = registry.factory("ssb_mod").unwrap();
    assert_eq!(ssb.kernels(), ["interleave", "rotate", "fir"]);
    assert_eq!(ssb.description().kind, "ssb_mod");
}

#[test]
fn costs_delays_and_memory_follow_the_table() {
    for (ty, precision) in [("f32", "f32"), ("q15", "q15")] {
        // am_mod: cost 2, no delay, no memory of its own.
        let report = validate(&single("am_mod", json!({}), ty, precision, 48_000));
        assert!(report.is_ok(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 2.0);
        let node = report.nodes.iter().find(|n| n.name == "n").unwrap();
        assert_eq!(node.rate_out, node.rate_in);
        assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO.reduced());

        // ssb_mod: cost 8 + 4 taps, delay (taps - 1) / 2.
        for (params, taps) in [
            (json!({}), 873u64),
            (
                json!({ "low_hz": 0, "high_hz": 10000, "transition_hz": 1000 }),
                SsbModConfig {
                    low_hz: 0.0,
                    high_hz: 10_000.0,
                    transition_hz: 1_000.0,
                    ..SsbModConfig::default()
                }
                .num_taps(sdrk::Rate::hz(48_000).unwrap())
                .unwrap() as u64,
            ),
        ] {
            let report = validate(&single("ssb_mod", params.clone(), ty, precision, 48_000));
            assert!(report.is_ok(), "{params}: {}", report.to_json());
            assert_eq!(report.cost_per_input_sample, 8.0 + 4.0 * taps as f64);
            let node = report.nodes.iter().find(|n| n.name == "n").unwrap();
            assert_eq!(node.rate_out, node.rate_in);
            assert_eq!(
                node.delay_samples_in.reduced(),
                Delay::new(taps - 1, 2).unwrap().reduced()
            );
        }
    }
}

#[test]
fn types_and_parameters_are_checked() {
    let code = |doc: &Value| validate(doc).errors.first().map(|d| d.code);
    // Both kinds take real audio only.
    for kind in ["am_mod", "ssb_mod"] {
        for (ty, precision) in [("c32", "f32"), ("cq15", "q15")] {
            assert_eq!(
                code(&single(kind, json!({}), ty, precision, 48_000)),
                Some(DiagCode::TypeMismatch),
                "{kind} {ty}"
            );
        }
    }
    // Ranges and the band rules of ssb_demod.
    for (kind, params) in [
        ("am_mod", json!({ "carrier": 1.5 })),
        ("am_mod", json!({ "carrier": -0.1 })),
        ("am_mod", json!({ "depth": 1.01 })),
        ("ssb_mod", json!({ "low_hz": 3000 })),
        ("ssb_mod", json!({ "high_hz": 23950 })),
        ("ssb_mod", json!({ "sideband": "dsb" })),
        ("ssb_mod", json!({ "max_taps": 100 })),
        ("ssb_mod", json!({ "attenuation_db": 5 })),
    ] {
        assert!(
            !validate(&single(kind, params.clone(), "f32", "f32", 48_000)).is_ok(),
            "{kind} {params}"
        );
    }
    // The band edge depends on the rate: 3000 Hz plus 100 Hz fits 8 kHz, 3950 Hz does not.
    assert!(validate(&single("ssb_mod", json!({}), "f32", "f32", 8_000)).is_ok());
    assert!(
        !validate(&single(
            "ssb_mod",
            json!({ "high_hz": 3950 }),
            "f32",
            "f32",
            8_000
        ))
        .is_ok()
    );
}

/// The runtime parameters (`depth`, `sideband`) can be set on a built pipeline; the build
/// parameters cannot.
#[test]
fn runtime_parameters_can_be_set_on_the_pipeline() {
    let mut p = build(&single("am_mod", json!({}), "f32", "f32", 48_000).to_string());
    p.set_param("n", "depth", json!(1.0)).unwrap();
    assert!(p.set_param("n", "depth", json!(1.5)).is_err());
    assert!(p.set_param("n", "carrier", json!(0.4)).is_err());
    let mut p = build(&single("ssb_mod", json!({}), "q15", "q15", 48_000).to_string());
    p.set_param("n", "sideband", json!("lsb")).unwrap();
    assert!(p.set_param("n", "sideband", json!("am")).is_err());
    assert!(p.set_param("n", "low_hz", json!(200)).is_err());
}

/// Pushes audio in pieces into a one-node pipeline and reads the baseband; the result equals
/// the block run on the whole audio.
fn push_equals_block<T: BlockSample + sdrk::Sample + PartialEq + core::fmt::Debug, B>(
    kind: &str,
    params: Value,
    precision: &str,
    make: impl Fn() -> B,
    wrap: impl Fn(&[T::Real]) -> InputData<'_>,
) where
    B: Block<In = T::Real, Out = T>,
{
    let ty = precision;
    let mut rng = Pcg32::new(483, 41);
    let audio: Vec<T::Real> = (0..6000)
        .map(|i| {
            let p = std::f64::consts::TAU * 1000.0 * i as f64 / 48_000.0;
            <T::Real as RealSample>::from_f64_round(0.5 * p.cos() + 0.05 * rng.normal())
        })
        .collect();
    let mut block = make();
    let mut want = vec![T::ZERO; audio.len()];
    let progress = block.process(&audio, &mut want).unwrap();
    assert_eq!(progress.produced, audio.len());

    let mut p = build(&single(kind, params, ty, precision, 48_000).to_string());
    let mut got = Vec::new();
    let mut buf = vec![T::ZERO; 4096];
    for piece in audio.chunks(777) {
        p.push("in", wrap(piece)).unwrap();
        loop {
            let n = p.read_stream("out", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            got.extend_from_slice(&buf[..n]);
        }
    }
    assert_eq!(got.len(), want.len(), "{kind} {precision}");
    assert!(
        got == want,
        "{kind} {precision}: the pipeline differs from the block"
    );
}

#[test]
fn a_push_run_equals_the_block() {
    let r = sdrk::Rate::hz(48_000).unwrap();
    push_equals_block::<C32, _>(
        "am_mod",
        json!({ "depth": 0.9 }),
        "f32",
        || {
            let cfg = AmModConfig {
                depth: 0.9,
                ..AmModConfig::default()
            };
            AmModBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap()
        },
        |x: &[f32]| InputData::F32(x),
    );
    push_equals_block::<CQ15, _>(
        "am_mod",
        json!({}),
        "q15",
        || AmModBlock::<CQ15, _>::new(&AmModConfig::default(), r, &mut VecStorage).unwrap(),
        |x: &[Q15]| InputData::Q15(x),
    );
    push_equals_block::<C32, _>(
        "ssb_mod",
        json!({ "sideband": "lsb" }),
        "f32",
        || {
            let cfg = SsbModConfig {
                sideband: SsbSideband::Lsb,
                ..SsbModConfig::default()
            };
            SsbModBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap()
        },
        |x: &[f32]| InputData::F32(x),
    );
    push_equals_block::<CQ15, _>(
        "ssb_mod",
        json!({}),
        "q15",
        || SsbModBlock::<CQ15, _>::new(&SsbModConfig::default(), r, &mut VecStorage).unwrap(),
        |x: &[Q15]| InputData::Q15(x),
    );
}
