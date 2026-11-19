//! The kind `scramble` (M49 §6, test 6): registration with the kernel `scramble`, the cost
//! `7 + 1.25 bits` (`bits` 1 and 2) and `10 + 1.25 bits` (`bits` 3 to 8, measured again in
//! M49.1 §6), the rate and delay 0, complex input only, the errors at their parameter (`bits` 0 and 9,
//! `key_seed` 0, a seed that masks to 0 as `invalid_value`, a key that is not a choice), the
//! schema, and push-mode flows `scramble` -> `scramble` (`inverse`) that equal the blocks bit for
//! bit in pieces of 1 to 1000 samples (seeds 696, 697), `c32` and `cq15`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::large_stack_frames,
    clippy::needless_pass_by_value
)]

mod common;

use sdrk::{C32, CQ15, Delay, Q15, Rate};
use sdrk_blocks::{Block, ScrambleBlock, ScrambleConfig};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Registry, VecStorage};
use sdrk_proto::prbs::Poly;
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::build;

/// The seeds of the signals and of the pieces (M49 test 6).
const SEEDS: [u64; 2] = [696, 697];

/// A one-node flow of `kind` on an input of type `ty` at `rate` Hz.
fn single(ty: &str, precision: &str, rate: u64, params: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": precision, "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": rate } },
        "nodes": { "n": { "kind": "scramble", "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

/// The flow `scramble` -> `scramble` (`inverse`) with an output after each node.
fn chain(ty: &str, precision: &str, params: &Value) -> Value {
    let mut inverse = params.clone();
    inverse["inverse"] = json!(true);
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": precision, "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": 48000 } },
        "nodes": {
            "tx": { "kind": "scramble", "params": params },
            "rx": { "kind": "scramble", "params": inverse }
        },
        "edges": [
            { "from": "in", "to": "tx" },
            { "from": "tx", "to": "rx" },
            { "from": "tx", "to": "mid" },
            { "from": "rx", "to": "out" }
        ],
        "outputs": { "mid": { "type": "stream" }, "out": { "type": "stream" } }
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
fn the_kind_is_registered_with_its_kernel() {
    let registry = Registry::standard();
    let f = registry.factory("scramble").unwrap();
    assert_eq!(f.kernels(), ["scramble"]);
    assert_eq!(f.description().kind, "scramble");
    assert_eq!(registry.descriptions().count(), 83);
    // The level is pinned by the facade test; here it only has to be a real implementation,
    // the same for both types.
    let level = sdrk::backend_for("scramble", sdrk::SampleType::C32);
    assert!(matches!(level, "simd" | "fallback"), "{level}");
    assert_eq!(sdrk::backend_for("scramble", sdrk::SampleType::CQ15), level);
}

#[test]
fn the_cost_is_the_kernel_plus_a_quarter_more_than_one_per_key_bit() {
    for (ty, precision) in [("c32", "f32"), ("cq15", "q15")] {
        for bits in 1..=8 {
            let report = validate(&single(ty, precision, 48_000, json!({ "bits": bits })));
            assert!(report.is_ok(), "{ty}: {}", report.to_json());
            // M49.1 §6: the kernel 7 (`bits` ≤ 2) or 10 (one product), the key 1.25 per bit.
            let kernel = if bits <= 2 { 7.0 } else { 10.0 };
            let cost = kernel + 1.25 * f64::from(bits);
            assert_eq!(report.cost_per_input_sample, cost, "{ty} bits {bits}");
            let node = &report.nodes[0];
            assert_eq!(node.rate_out, Rate::hz(48_000).unwrap(), "{ty}");
            assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO, "{ty}");
        }
    }
    // Every parameter has a default.
    assert!(validate(&single("c32", "f32", 8_000, json!({}))).is_ok());
}

#[test]
fn a_real_input_is_a_type_mismatch() {
    for (ty, precision) in [("f32", "f32"), ("q15", "q15")] {
        let report = validate(&single(ty, precision, 48_000, json!({})));
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
        (
            json!({ "bits": 0 }),
            "nodes.n.params.bits",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "bits": 9 }),
            "nodes.n.params.bits",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "bits": "two" }),
            "nodes.n.params.bits",
            DiagCode::WrongType,
        ),
        (
            json!({ "key_seed": 0 }),
            "nodes.n.params.key_seed",
            DiagCode::OutOfRange,
        ),
        (
            json!({ "key_seed": 32768 }),
            "nodes.n.params.key_seed",
            DiagCode::OutOfRange,
        ),
        // In range, but the low 5 bits of 32 are zero: no state for prbs5.
        (
            json!({ "key": "prbs5", "key_seed": 32 }),
            "nodes.n.params.key_seed",
            DiagCode::InvalidValue,
        ),
        (
            json!({ "key": "prbs11", "key_seed": 4096 }),
            "nodes.n.params.key_seed",
            DiagCode::InvalidValue,
        ),
        (
            json!({ "inverse": 1 }),
            "nodes.n.params.inverse",
            DiagCode::WrongType,
        ),
    ] {
        let (p, c) = first_error(&single("c32", "f32", 48_000, params.clone()));
        assert_eq!((p.as_str(), c), (path, code), "{params}");
    }
    // A key that is not one of the choices is an error at `key`.
    let (p, _) = first_error(&single("c32", "f32", 48_000, json!({ "key": "prbs23" })));
    assert_eq!(p, "nodes.n.params.key");
    // The limits and every key are valid.
    for key in ["prbs5", "prbs7", "prbs9", "prbs10", "prbs11", "prbs15"] {
        assert!(
            validate(&single(
                "cq15",
                "q15",
                48_000,
                json!({ "bits": 8, "key": key, "key_seed": 32767, "inverse": true }),
            ))
            .is_ok(),
            "{key}"
        );
    }
}

#[test]
fn the_schema_has_the_kind() {
    let schema = Registry::standard().json_schema_text();
    assert!(schema.contains("\"const\": \"scramble\""));
}

/// Pushes random pieces (1 to 1000 samples) and reads both outputs after each push.
fn push<T, F>(
    p: &mut sdrk_pipeline::Pipeline,
    input: &[T],
    wrap: F,
    rng: &mut Pcg32,
) -> (Vec<T>, Vec<T>)
where
    T: Copy + Default + sdrk::Sample,
    F: for<'a> Fn(&'a [T]) -> InputData<'a>,
{
    let (mut mid, mut out) = (Vec::new(), Vec::new());
    let mut buf = vec![T::default(); 4096];
    let mut rest = input;
    while !rest.is_empty() {
        let n = (1 + rng.next_u32() as usize % 1000).min(rest.len());
        p.push("in", wrap(&rest[..n])).unwrap();
        rest = &rest[n..];
        let k = p.read_stream::<T>("mid", &mut buf).unwrap();
        mid.extend_from_slice(&buf[..k]);
        let k = p.read_stream::<T>("out", &mut buf).unwrap();
        out.extend_from_slice(&buf[..k]);
    }
    (mid, out)
}

/// The blocks wired by hand, whole buffers in pieces of `SCRAMBLE_CHUNK`.
fn blocks<T>(cfg: &ScrambleConfig, input: &[T]) -> (Vec<T>, Vec<T>)
where
    T: sdrk_blocks::BlockSample + Copy + Default,
    sdrk::Sdrk: sdrk_core::kernels::scramble::Scramble<T>,
{
    let r = Rate::hz(48_000).unwrap();
    let mut tx = ScrambleBlock::<T, VecStorage>::new(cfg, r, &mut VecStorage).unwrap();
    let inv = ScrambleConfig {
        inverse: true,
        ..*cfg
    };
    let mut rx = ScrambleBlock::<T, VecStorage>::new(&inv, r, &mut VecStorage).unwrap();
    let run = |b: &mut ScrambleBlock<T, VecStorage>, x: &[T]| {
        let mut y = vec![T::default(); x.len()];
        let mut done = 0;
        while done < x.len() {
            let p = b.process(&x[done..], &mut y[done..]).unwrap();
            assert!(p.consumed > 0);
            done += p.consumed;
        }
        y
    };
    let mid = run(&mut tx, input);
    let out = run(&mut rx, &mid);
    (mid, out)
}

fn bytes<T: bytemuck::Pod>(v: &[T]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

/// The configurations of the flows: one per key, `bits` 1 to 8 between them.
fn configs() -> Vec<(ScrambleConfig, Value)> {
    [
        (Poly::Prbs5, "prbs5", 1, 3),
        (Poly::Prbs7, "prbs7", 2, 100),
        (Poly::Prbs9, "prbs9", 3, 511),
        (Poly::Prbs10, "prbs10", 5, 1023),
        (Poly::Prbs11, "prbs11", 7, 1),
        (Poly::Prbs15, "prbs15", 8, 32767),
        (Poly::Prbs15, "prbs15", 4, 12345),
        (Poly::Prbs9, "prbs9", 6, 77),
    ]
    .into_iter()
    .map(|(key, name, bits, key_seed)| {
        (
            ScrambleConfig {
                bits,
                inverse: false,
                key,
                key_seed,
            },
            json!({ "bits": bits, "key": name, "key_seed": key_seed }),
        )
    })
    .collect()
}

#[test]
fn a_c32_flow_equals_the_blocks() {
    let n = 6_000;
    for seed in SEEDS {
        let mut rng = Pcg32::new(seed, 1);
        let input: Vec<C32> = (0..n)
            .map(|_| C32::new(rng.normal() as f32 * 0.3, rng.normal() as f32 * 0.3))
            .collect();
        for (cfg, params) in configs() {
            let (want_mid, want_out) = blocks(&cfg, &input);
            let mut p = build(&chain("c32", "f32", &params).to_string());
            let (mid, out) = push(&mut p, &input, |x| InputData::C32(x), &mut rng);
            assert_eq!(bytes(&mid), bytes(&want_mid), "{params} seed {seed}: mid");
            assert_eq!(bytes(&out), bytes(&want_out), "{params} seed {seed}: out");
            if cfg.bits <= 2 {
                assert_eq!(bytes(&out), bytes(&input), "{params}: exact round trip");
            }
        }
    }
}

#[test]
fn a_cq15_flow_equals_the_blocks() {
    let n = 6_000;
    for seed in SEEDS {
        let mut rng = Pcg32::new(seed, 2);
        let input: Vec<CQ15> = (0..n)
            .map(|_| CQ15 {
                re: Q15((rng.next_u32() >> 17) as i16 - 16384),
                im: Q15((rng.next_u32() >> 17) as i16 - 16384),
            })
            .collect();
        for (cfg, params) in configs() {
            let (want_mid, want_out) = blocks(&cfg, &input);
            let mut p = build(&chain("cq15", "q15", &params).to_string());
            let (mid, out) = push(&mut p, &input, |x| InputData::CQ15(x), &mut rng);
            assert_eq!(mid, want_mid, "{params} seed {seed}: mid");
            assert_eq!(out, want_out, "{params} seed {seed}: out");
            if cfg.bits <= 2 {
                assert_eq!(out, input, "{params}: exact round trip");
            }
        }
    }
}
