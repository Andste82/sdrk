//! The kind `correlate` (M50 §6, test 7; seed 707): registration with the kernel `xcorr_fft`,
//! the cost `(5 N log2 N + 4 N) / B` per input sample, the rate, the delay `L - 1` and the
//! `min_input` `B`, `c32` input only, the errors at their parameter (the FFT size too small for
//! the code as `invalid_value` at `params.fft_size` with the required size in the message), the
//! schema, and push-mode flows that equal the block bit for bit in pieces of 1 to 1000 samples,
//! also with a block size above `max_chunk`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::large_stack_frames
)]

mod common;

use sdrk::{C32, Delay, Rate};
use sdrk_blocks::{Block, CorrelateBlock, CorrelateConfig};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry, VecStorage};
use sdrk_proto::prbs::Poly;
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::build;

/// The seed of the signals and of the pieces (M50 test 7 of the kind).
const SEED: u64 = 707;

/// A one-node flow `in` -> `correlate` -> `out` on an input of type `ty`.
fn single(ty: &str, precision: &str, max_chunk: usize, params: &Value) -> Value {
    json!({
        "sdrk": 1, "name": "t",
        "settings": { "precision": precision, "max_chunk": max_chunk },
        "inputs": { "in": { "type": ty, "rate_hz": 48000 } },
        "nodes": { "n": { "kind": "correlate", "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

fn validate(doc: &Value) -> sdrk_pipeline::Report {
    Registry::standard().validate_json(&doc.to_string(), &BuildOptions::default())
}

fn first_error(doc: &Value) -> (String, DiagCode, String) {
    let report = validate(doc);
    let e = report.errors.first().expect("an error");
    (e.path.clone(), e.code, e.message.clone())
}

#[test]
fn the_kind_is_registered_with_its_kernel() {
    let registry = Registry::standard();
    let f = registry.factory("correlate").unwrap();
    assert_eq!(f.kernels(), ["xcorr_fft"]);
    assert_eq!(f.description().kind, "correlate");
    assert_eq!(registry.descriptions().count(), 85);
    let level = sdrk::backend_for("xcorr_fft", sdrk::SampleType::C32);
    assert!(matches!(level, "simd" | "fallback"), "{level}");
}

#[test]
fn the_cost_is_two_ffts_and_the_product_per_block() {
    for (params, l, n) in [
        (json!({}), 127, 512),
        (json!({ "code": "prbs5" }), 31, 128),
        (
            json!({ "code": "prbs10", "samples_per_chip": 2 }),
            2046,
            8192,
        ),
        (
            json!({ "code": "prbs10", "samples_per_chip": 2, "fft_size": 4096 }),
            2046,
            4096,
        ),
        (json!({ "code": "prbs15" }), 32_767, 65_536),
    ] {
        let report = validate(&single("c32", "f32", 4096, &params));
        assert!(report.is_ok(), "{params}: {}", report.to_json());
        let (nf, b) = (n as f64, (n - l + 1) as f64);
        let cost = (5.0 * nf * nf.log2() + 4.0 * nf) / b;
        let got = report.cost_per_input_sample;
        assert!(
            (got - cost).abs() <= 1e-12 * cost,
            "{params}: {got} against {cost}"
        );
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::hz(48_000).unwrap(), "{params}");
        assert_eq!(
            node.delay_samples_in.reduced(),
            Delay::new(l as u64 - 1, 1).unwrap(),
            "{params}"
        );
    }
    // The trade-off of the documentation: the smallest N costs about 1.4 times the automatic.
    let cost = |n: f64, l: f64| (5.0 * n * n.log2() + 4.0 * n) / (n - l + 1.0);
    let ratio = cost(4096.0, 2046.0) / cost(8192.0, 2046.0);
    assert!((1.35..1.45).contains(&ratio), "{ratio}");
}

#[test]
fn a_complex_q15_or_real_input_is_a_type_mismatch() {
    for (ty, precision) in [("cq15", "q15"), ("f32", "f32"), ("q15", "q15")] {
        let report = validate(&single(ty, precision, 1024, &json!({})));
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.code == DiagCode::TypeMismatch),
            "{ty}: {}",
            report.to_json()
        );
    }
}

#[test]
fn invalid_parameters_are_errors_at_their_parameter() {
    for (params, path, code) in [
        (json!({ "seed": 0 }), "seed", DiagCode::OutOfRange),
        (json!({ "seed": 32768 }), "seed", DiagCode::OutOfRange),
        // In range, but the low 5 bits of 32 are zero: no state for prbs5.
        (
            json!({ "code": "prbs5", "seed": 32 }),
            "seed",
            DiagCode::InvalidValue,
        ),
        (
            json!({ "samples_per_chip": 0 }),
            "samples_per_chip",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "samples_per_chip": 17 }),
            "samples_per_chip",
            DiagCode::OutOfRange,
        ),
        (json!({ "fft_size": -1 }), "fft_size", DiagCode::OutOfRange),
        (
            json!({ "fft_size": 131_072 }),
            "fft_size",
            DiagCode::OutOfRange,
        ),
        (json!({ "fft_size": 8 }), "fft_size", DiagCode::InvalidValue),
        (
            json!({ "fft_size": 1000 }),
            "fft_size",
            DiagCode::InvalidValue,
        ),
        (
            json!({ "samples_per_chip": "two" }),
            "samples_per_chip",
            DiagCode::WrongType,
        ),
    ] {
        let (p, c, _) = first_error(&single("c32", "f32", 1024, &params));
        assert_eq!(
            (p.as_str(), c),
            (format!("nodes.n.params.{path}").as_str(), code),
            "{params}"
        );
    }
    let (p, _, _) = first_error(&single("c32", "f32", 1024, &json!({ "code": "prbs23" })));
    assert_eq!(p, "nodes.n.params.code");
    // A size below 2 L names the required size.
    let (p, c, m) = first_error(&single("c32", "f32", 1024, &json!({ "fft_size": 128 })));
    assert_eq!(
        (p.as_str(), c),
        ("nodes.n.params.fft_size", DiagCode::InvalidValue)
    );
    assert!(
        m.contains("fft_size 256 or more") && m.contains("got 128"),
        "{m}"
    );
    // A code longer than 32768 samples needs more than the largest size.
    let (p, c, m) = first_error(&single(
        "c32",
        "f32",
        1024,
        &json!({ "code": "prbs15", "samples_per_chip": 2 }),
    ));
    assert_eq!(
        (p.as_str(), c),
        ("nodes.n.params.fft_size", DiagCode::InvalidValue)
    );
    assert!(m.contains("131072") && m.contains("largest 65536"), "{m}");
    // Every code and the limits are valid.
    for code in ["prbs5", "prbs7", "prbs9", "prbs10", "prbs11", "prbs15"] {
        assert!(
            validate(&single(
                "c32",
                "f32",
                1024,
                &json!({ "code": code, "seed": 32767 })
            ))
            .is_ok(),
            "{code}"
        );
    }
    assert!(
        validate(&single(
            "c32",
            "f32",
            1024,
            &json!({ "code": "prbs11", "samples_per_chip": 16, "fft_size": 65536 }),
        ))
        .is_ok()
    );
}

#[test]
fn the_schema_has_the_kind() {
    let schema = Registry::standard().json_schema_text();
    assert!(schema.contains("\"const\": \"correlate\""));
}

/// Pushes random pieces (1 to 1000 samples) and reads the whole output after each push.
fn push(p: &mut sdrk_pipeline::Pipeline, input: &[C32], rng: &mut Pcg32) -> Vec<C32> {
    let mut out = Vec::new();
    let mut buf = vec![C32::default(); 1 << 17];
    let mut rest = input;
    while !rest.is_empty() {
        let n = (1 + rng.next_u32() as usize % 1000).min(rest.len());
        p.push("in", InputData::C32(&rest[..n])).unwrap();
        rest = &rest[n..];
        // The node writes whole blocks of `B` samples, more than one push estimates: read
        // until the output is empty (reading continues the processing, back-pressure).
        loop {
            let k = p.read_stream::<C32>("out", &mut buf).unwrap();
            if k == 0 {
                break;
            }
            out.extend_from_slice(&buf[..k]);
        }
    }
    out
}

/// The block over the whole input in one call.
fn block(cfg: &CorrelateConfig, input: &[C32]) -> Vec<C32> {
    let r = Rate::hz(48_000).unwrap();
    let mut b = CorrelateBlock::<VecStorage>::new(cfg, r, &mut VecStorage).unwrap();
    let mut y = vec![C32::default(); input.len()];
    let p = b.process(input, &mut y).unwrap();
    y.truncate(p.produced);
    y
}

fn bits(v: &[C32]) -> Vec<(u32, u32)> {
    v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

#[test]
fn a_flow_equals_the_block() {
    let mut rng = Pcg32::new(SEED, 1);
    // The block size of the last two is above `max_chunk` (6147 and 32770 against 1024).
    for (cfg, params, max_chunk, n) in [
        (CorrelateConfig::default(), json!({}), 1024, 6_000),
        (
            CorrelateConfig {
                code: Poly::Prbs5,
                seed: 9,
                samples_per_chip: 3,
                fft_size: 512,
            },
            json!({ "code": "prbs5", "seed": 9, "samples_per_chip": 3, "fft_size": 512 }),
            1024,
            6_000,
        ),
        (
            CorrelateConfig {
                code: Poly::Prbs10,
                samples_per_chip: 2,
                ..CorrelateConfig::default()
            },
            json!({ "code": "prbs10", "samples_per_chip": 2 }),
            1024,
            20_000,
        ),
        (
            CorrelateConfig {
                code: Poly::Prbs15,
                ..CorrelateConfig::default()
            },
            json!({ "code": "prbs15" }),
            1024,
            70_000,
        ),
    ] {
        let input: Vec<C32> = (0..n)
            .map(|_| C32::new(rng.normal() as f32 * 0.3, rng.normal() as f32 * 0.3))
            .collect();
        let want = block(&cfg, &input);
        let b = cfg.resolved_fft_size().unwrap() - cfg.reference_len().unwrap() + 1;
        assert_eq!(want.len(), n / b * b, "{params}");
        let mut p = build(&single("c32", "f32", max_chunk, &params).to_string());
        let got = push(&mut p, &input, &mut rng);
        assert_eq!(bits(&got), bits(&want), "{params}");
    }
}
