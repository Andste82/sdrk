//! The kinds of M39: `fsk_mod`, `fsk_demod` and `afc` (registration with their kernels, the
//! costs `8 + span sps`, `12 + round(sps) + 24 / sps` and 20, rates and delays, type checks,
//! `rate_unsupported` below 4 samples per symbol), the constellation `fsk4` of `slice`, the
//! readings of `afc` in the statistics and its runtime `hold`, and a push-mode chain `afc` ->
//! `fsk_demod` that equals the blocks bit for bit.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::large_stack_frames,
    clippy::needless_pass_by_value,
    clippy::assert_is_empty
)]

mod common;

use sdrk::{C32, Delay, Rate};
use sdrk_blocks::{
    AfcBlock, AfcConfig, Block, FskDemodBlock, FskDemodConfig, FskModBlock, FskModConfig,
};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry, VecStorage};
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::build;

/// A one-node flow of `kind` on an input of type `ty` at `rate` Hz.
fn single(kind: &str, ty: &str, rate: u64, params: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": rate } },
        "nodes": { "n": { "kind": kind, "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

fn validate(doc: &Value) -> sdrk_pipeline::Report {
    Registry::standard().validate_json(&doc.to_string(), &BuildOptions::default())
}

fn first_error(doc: &Value) -> (String, DiagCode) {
    let report = validate(doc);
    let e = report.errors.first().expect("an error");
    (e.path.clone(), e.code)
}

#[test]
fn the_kinds_are_registered_with_their_kernels() {
    let registry = Registry::standard();
    for (kind, kernels) in [
        ("fsk_mod", &["fir_interpolate_real", "fm_mod"][..]),
        (
            "fsk_demod",
            &["quad_demod", "resample_real", "symbol_sync"][..],
        ),
        ("afc", &["rotate", "quad_demod"][..]),
    ] {
        let f = registry.factory(kind).unwrap();
        assert_eq!(f.kernels(), kernels, "{kind}");
        assert_eq!(f.description().kind, kind);
    }
    assert_eq!(registry.descriptions().count(), 65);
}

#[test]
fn fsk_mod_costs_8_plus_span_sps() {
    for (levels, sps, bt, span, rate_out, delay) in [
        (2u64, 4u64, 0.5, 4u64, 200_000u64, (19u64, 8u64)),
        (2, 8, 0.0, 4, 400_000, (7, 16)),
        (4, 8, 0.5, 2, 200_000, (46, 16)),
    ] {
        let report = validate(&single(
            "fsk_mod",
            "f32",
            50_000,
            json!({ "levels": levels, "samples_per_symbol": sps, "deviation_hz": 25_000,
                    "bt": bt, "span": span }),
        ));
        assert!(report.is_ok(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 8.0 + (span * sps) as f64);
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::hz(rate_out).unwrap());
        assert_eq!(
            node.delay_samples_in.reduced(),
            Delay::new(delay.0, delay.1).unwrap().reduced()
        );
    }
    // Required parameters, ranges and types.
    assert!(
        !validate(&single(
            "fsk_mod",
            "f32",
            50_000,
            json!({ "samples_per_symbol": 4 })
        ))
        .is_ok()
    );
    assert!(
        !validate(&single(
            "fsk_mod",
            "f32",
            50_000,
            json!({ "deviation_hz": 1000 })
        ))
        .is_ok()
    );
    for bad in [
        json!({ "samples_per_symbol": 4, "deviation_hz": 1000, "levels": 3 }),
        json!({ "samples_per_symbol": 4, "deviation_hz": 1000, "bt": 0.1 }),
        json!({ "samples_per_symbol": 65, "deviation_hz": 1000 }),
        json!({ "samples_per_symbol": 4, "deviation_hz": 100_001 }),
    ] {
        assert!(
            !validate(&single("fsk_mod", "f32", 50_000, bad.clone())).is_ok(),
            "{bad} must be rejected"
        );
    }
    for ty in ["c32", "q15", "cq15"] {
        let (_, code) = first_error(&single(
            "fsk_mod",
            ty,
            50_000,
            json!({ "samples_per_symbol": 4, "deviation_hz": 1000 }),
        ));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
}

#[test]
fn fsk_demod_costs_12_plus_round_sps_plus_24_per_symbol_and_needs_4_samples_per_symbol() {
    for (levels, fs, sym, sps, rate_out) in [
        (
            2u64,
            200_000u64,
            50_000.0,
            4.0f64,
            Rate::hz(50_000).unwrap(),
        ),
        (4, 48_000, 4_800.0, 10.0, Rate::hz(9_600).unwrap()),
        (2, 10_100, 1_000.0, 10.1, Rate::hz(1_000).unwrap()),
    ] {
        let report = validate(&single(
            "fsk_demod",
            "c32",
            fs,
            json!({ "levels": levels, "symbol_rate_hz": sym, "deviation_hz": sym / 2.0 }),
        ));
        assert!(report.is_ok(), "{}", report.to_json());
        assert_eq!(
            report.cost_per_input_sample,
            12.0 + sps.round() + 24.0 / sps
        );
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, rate_out);
        let taps = sps.round() as u64;
        assert_eq!(
            node.delay_samples_in.reduced(),
            Delay::new(taps + 2, 2).unwrap().reduced()
        );
    }
    let (path, code) = first_error(&single(
        "fsk_demod",
        "c32",
        199_999,
        json!({ "symbol_rate_hz": 50_000, "deviation_hz": 25_000 }),
    ));
    assert_eq!(code, DiagCode::RateUnsupported);
    assert_eq!(path, "nodes.n.params.symbol_rate_hz");
    assert!(
        !validate(&single(
            "fsk_demod",
            "c32",
            200_000,
            json!({ "symbol_rate_hz": 50_000 })
        ))
        .is_ok()
    );
    for ty in ["f32", "cq15"] {
        let (_, code) = first_error(&single(
            "fsk_demod",
            ty,
            200_000,
            json!({ "symbol_rate_hz": 50_000, "deviation_hz": 25_000 }),
        ));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
}

#[test]
fn afc_costs_20_and_slice_takes_fsk4() {
    let report = validate(&single(
        "afc",
        "c32",
        200_000,
        json!({ "max_offset_hz": 20_000, "carrier_hz": 868e6 }),
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.cost_per_input_sample, 20.0);
    assert_eq!(report.nodes[0].rate_out, Rate::hz(200_000).unwrap());
    assert_eq!(
        report.nodes[0].delay_samples_in.reduced(),
        Delay::ZERO.reduced()
    );
    assert!(!validate(&single("afc", "c32", 200_000, json!({}))).is_ok());
    let (path, code) = first_error(&single(
        "afc",
        "c32",
        200_000,
        json!({ "max_offset_hz": 100_000 }),
    ));
    assert_eq!(code, DiagCode::RateUnsupported);
    assert_eq!(path, "nodes.n.params.max_offset_hz");
    let (_, code) = first_error(&single(
        "afc",
        "f32",
        200_000,
        json!({ "max_offset_hz": 1000 }),
    ));
    assert_eq!(code, DiagCode::TypeMismatch);
    // fsk4 takes f32 and gives two soft bits per symbol.
    let report = validate(&single(
        "slice",
        "f32",
        4800,
        json!({ "constellation": "fsk4" }),
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.nodes[0].rate_out, Rate::hz(9600).unwrap());
    let (path, code) = first_error(&single(
        "slice",
        "c32",
        4800,
        json!({ "constellation": "fsk4" }),
    ));
    assert_eq!(code, DiagCode::InvalidValue);
    assert_eq!(path, "nodes.n.params.constellation");
}

/// The receive chain `afc` -> `fsk_demod` at 80 kHz (10000 symbols per second); the test
/// modulates with the `fsk_mod` block and adds the carrier offset outside the flow.
fn rx_chain() -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": "c32", "rate_hz": 80_000 } },
        "nodes": {
            "afc": { "kind": "afc", "params": { "max_offset_hz": 5000, "time_constant_s": 0.05,
                     "carrier_hz": 433.92e6 } },
            "demod": { "kind": "fsk_demod", "params": { "symbol_rate_hz": 10_000,
                       "deviation_hz": 5000 } }
        },
        "edges": [
            { "from": "in", "to": "afc" },
            { "from": "afc", "to": "demod" },
            { "from": "demod", "to": "bits" }
        ],
        "outputs": { "bits": { "type": "stream" } }
    })
}

#[test]
fn the_receive_chain_equals_the_blocks_and_reports_the_offset() {
    let fs = Rate::hz(80_000).unwrap();
    let mut rng = Pcg32::new(537, 60);
    let bits: Vec<f32> = (0..20_000)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect();
    let mod_cfg = FskModConfig {
        samples_per_symbol: 8,
        deviation_hz: 5_000.0,
        ..FskModConfig::default()
    };
    let mut fsk = FskModBlock::new(&mod_cfg, Rate::hz(10_000).unwrap(), &mut VecStorage).unwrap();
    let mut signal = vec![C32::default(); bits.len() * 8];
    fsk.process(&bits, &mut signal).unwrap();
    // A carrier offset of +1.5 kHz.
    for (i, s) in signal.iter_mut().enumerate() {
        let ph = std::f64::consts::TAU * 1_500.0 * i as f64 / 80_000.0;
        let (sin, cos) = ph.sin_cos();
        let (re, im) = (f64::from(s.re), f64::from(s.im));
        *s = C32::new((re * cos - im * sin) as f32, (re * sin + im * cos) as f32);
    }
    // The blocks directly.
    let afc_cfg = AfcConfig {
        max_offset_hz: 5_000.0,
        time_constant_s: 0.05,
        carrier_hz: 433.92e6,
        ..AfcConfig::default()
    };
    let mut afc = AfcBlock::new(&afc_cfg, fs, &mut VecStorage).unwrap();
    let mut shifted = vec![C32::default(); signal.len()];
    afc.process(&signal, &mut shifted).unwrap();
    let demod_cfg = FskDemodConfig {
        symbol_rate_hz: 10_000.0,
        deviation_hz: 5_000.0,
        ..FskDemodConfig::default()
    };
    let mut demod = FskDemodBlock::new(&demod_cfg, fs, &mut VecStorage).unwrap();
    let mut want = vec![0.0f32; bits.len() + 16];
    let p = demod.process(&shifted, &mut want).unwrap();
    assert_eq!(p.consumed, shifted.len());
    want.truncate(p.produced);
    // The pipeline in pieces of up to 1000 samples.
    let mut pipe = build(&rx_chain().to_string());
    let names: Vec<_> = pipe.stats()[0].readings.iter().map(|r| r.0).collect();
    assert_eq!(pipe.stats()[0].name, "afc");
    assert_eq!(names, ["offset_hz", "offset_ppm"]);
    assert_eq!(pipe.stats()[0].readings[0].1, 0.0);
    assert!(pipe.stats()[1].readings.is_empty());
    let mut got = Vec::new();
    let mut buf = vec![0.0f32; 4096];
    for piece in signal.chunks(1000) {
        pipe.push("in", InputData::C32(piece)).unwrap();
        let n = pipe.read_stream::<f32>("bits", &mut buf).unwrap();
        got.extend_from_slice(&buf[..n]);
    }
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "pipeline differs from the blocks"
    );
    // The readings of afc are those of the block.
    let st = &pipe.stats()[0];
    assert_eq!(st.readings[0], ("offset_hz", afc.offset_hz()));
    assert_eq!(st.readings[1], ("offset_ppm", afc.offset_ppm()));
    // With a time constant of 500 symbols the imbalance of the random data moves the
    // estimate by about deviation / sqrt(500), some 220 Hz.
    assert!(
        (afc.offset_hz() - 1_500.0).abs() < 500.0,
        "{}",
        afc.offset_hz()
    );
    // hold is a runtime parameter, the other parameters are build parameters.
    assert!(pipe.set_param("afc", "hold", json!(true)).is_ok());
    assert!(pipe.set_param("afc", "max_offset_hz", json!(1000)).is_err());
    assert!(pipe.set_param("demod", "loop_bw", json!(0.02)).is_err());
    // reset clears the estimate and its reading.
    pipe.reset();
    assert_eq!(pipe.stats()[0].readings[0], ("offset_hz", 0.0));
}
