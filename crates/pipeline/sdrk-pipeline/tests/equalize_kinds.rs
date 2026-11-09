//! The kind of M43 §2 and the options of §3: `equalize` (registration with its kernel, the cost
//! `8 (taps + feedback_taps)`, rate and delay `(taps - 1) / 2` symbols, `c32` only,
//! `invalid_value` for an even number of taps, the runtime parameters `mu_cma` and `mu_dd`, a
//! push-mode node equal to the block in pieces of seed 595) and `equalizer_taps` and
//! `equalizer_feedback_taps` of `psk_demod` (the kernel list with `equalize` before `costas`,
//! the cost `+ 8 (taps + feedback) / sps`, the delay, `invalid_value` for 1 and even counts and
//! for feedback taps without the equalizer, a push-mode node equal to the block).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::needless_pass_by_value,
    clippy::many_single_char_names,
    clippy::large_stack_frames
)]

mod common;

use sdrk::{C32, Delay, Rate};
use sdrk_blocks::{
    Block, EqConstellation, EqualizeBlock, EqualizeConfig, PskDemodBlock, PskDemodConfig,
    PskModBlock, PskModConfig,
};
use sdrk_pipeline::{BuildOptions, DiagCode, InputData, Pipeline, Registry, VecStorage};
use sdrk_testkit::Pcg32;
use serde_json::{Value, json};

use common::build;

/// Seed of the chunk plans and the random signals of M43 (test 6).
const SEED: u64 = 595;

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

/// Pushes `input` in pieces of 1 to 1024 samples (seed 595, stream `stream`) and reads the
/// stream output `out` after every push.
fn push_all<O: Copy + Default + sdrk::Sample>(
    p: &mut Pipeline,
    input: &[C32],
    stream: u64,
) -> Vec<O> {
    let mut rng = Pcg32::new(SEED, stream);
    let mut got = Vec::new();
    let mut buf = vec![O::default(); 1 << 16];
    let mut at = 0;
    while at < input.len() {
        let n = (1 + rng.below(1024) as usize).min(input.len() - at);
        p.push("in", InputData::C32(&input[at..at + n])).unwrap();
        at += n;
        loop {
            let k = p.read_stream::<O>("out", &mut buf).unwrap();
            if k == 0 {
                break;
            }
            got.extend_from_slice(&buf[..k]);
        }
    }
    got
}

/// Noisy QPSK symbols through `1 + 0.3 e^(j0.6) z^-1` with a constant phase.
fn symbols(n: usize, stream: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED, stream);
    let h = core::f32::consts::FRAC_1_SQRT_2;
    let s: Vec<C32> = (0..n)
        .map(|_| {
            let k = rng.next_u32();
            C32::new(
                if k & 1 == 0 { h } else { -h },
                if k & 2 == 0 { h } else { -h },
            )
        })
        .collect();
    let (c, d) = (0.3 * 0.6f32.cos(), 0.3 * 0.6f32.sin());
    let turn = C32::new(0.5f32.cos(), 0.5f32.sin());
    (0..n)
        .map(|k| {
            let mut y = s[k];
            if k > 0 {
                y += s[k - 1] * C32::new(c, d);
            }
            y * turn + C32::new(0.03 * rng.normal() as f32, 0.03 * rng.normal() as f32)
        })
        .collect()
}

#[test]
fn equalize_is_registered_with_its_kernel() {
    let registry = Registry::standard();
    let f = registry.factory("equalize").unwrap();
    assert_eq!(f.kernels(), ["equalize"]);
    assert_eq!(f.description().kind, "equalize");
    assert_eq!(registry.descriptions().count(), 68);
}

#[test]
fn equalize_costs_8_taps_and_keeps_the_rate() {
    for (taps, feedback) in [(3u64, 0u64), (11, 0), (63, 0), (15, 5)] {
        let report = validate(&single(
            "equalize",
            "c32",
            25_000,
            json!({ "taps": taps, "feedback_taps": feedback, "constellation": "8psk" }),
        ));
        assert!(report.is_ok(), "{}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 8.0 * (taps + feedback) as f64);
        let node = &report.nodes[0];
        assert_eq!(node.rate_out, Rate::hz(25_000).unwrap());
        assert_eq!(
            node.delay_samples_in.reduced(),
            Delay::new((taps - 1) / 2, 1).unwrap().reduced()
        );
    }
    let report = validate(&single("equalize", "c32", 25_000, json!({})));
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.cost_per_input_sample, 88.0);
    for ty in ["cq15", "f32", "q15"] {
        let (_, code) = first_error(&single("equalize", ty, 1000, json!({})));
        assert_eq!(code, DiagCode::TypeMismatch, "{ty}");
    }
    let (path, code) = first_error(&single("equalize", "c32", 1000, json!({ "taps": 10 })));
    assert_eq!(
        (path.as_str(), code),
        ("nodes.n.params.taps", DiagCode::InvalidValue)
    );
    for bad in [
        json!({ "taps": 1 }),
        json!({ "taps": 65 }),
        json!({ "mu_cma": 1.2 }),
        json!({ "mu_dd": 0 }),
        json!({ "feedback_taps": 64 }),
        json!({ "phase_bw": 0.00005 }),
        json!({ "phase_bw": 0.3 }),
        json!({ "constellation": "16qam" }),
        json!({ "cma_symbols": -1 }),
        json!({ "cma_symbols": 4_294_967_296u64 }),
    ] {
        assert!(
            !validate(&single("equalize", "c32", 1000, bad.clone())).is_ok(),
            "{bad}"
        );
    }
    let report = validate(&single(
        "equalize",
        "c32",
        1000,
        json!({ "cma_symbols": 4_294_967_295u64 }),
    ));
    assert!(report.is_ok(), "{}", report.to_json());
    // M44: the QAM constellations, at the cost of the taps (D989)
    for c in ["qam16", "qam64"] {
        let report = validate(&single(
            "equalize",
            "c32",
            25_000,
            json!({ "taps": 15, "constellation": c }),
        ));
        assert!(report.is_ok(), "{c}: {}", report.to_json());
        assert_eq!(report.cost_per_input_sample, 8.0 * 15.0, "{c}");
    }
}

/// Push mode in pieces equals the block, the runtime parameters change the node from the next
/// symbol, and the build parameters are read-only.
#[test]
fn equalize_kind_equals_the_block_and_takes_runtime_parameters() {
    let cfg = EqualizeConfig {
        taps: 15,
        constellation: EqConstellation::Qpsk,
        cma_symbols: 1500,
        ..EqualizeConfig::default()
    };
    let input = symbols(6000, 1);
    let fs = Rate::hz(25_000).unwrap();
    let mut block = EqualizeBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
    let mut want = vec![C32::default(); input.len()];
    let p = block.process(&input, &mut want).unwrap();
    assert_eq!(p.produced, input.len());
    let params = json!({ "taps": 15, "constellation": "qpsk", "cma_symbols": 1500 });
    let mut node = build(&single("equalize", "c32", 25_000, params.clone()).to_string());
    let got: Vec<C32> = push_all(&mut node, &input, 2);
    assert_eq!(got, want, "equalize pipeline differs from the block");

    // runtime parameters after 3000 symbols, as on the block
    let mut block = EqualizeBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
    let mut want = vec![C32::default(); input.len()];
    block.process(&input[..3000], &mut want[..3000]).unwrap();
    block
        .set_param("mu_dd", sdrk_blocks::ParamValue::F64(0.002))
        .unwrap();
    block
        .set_param("mu_cma", sdrk_blocks::ParamValue::F64(0.0001))
        .unwrap();
    block.process(&input[3000..], &mut want[3000..]).unwrap();
    let mut node = build(&single("equalize", "c32", 25_000, params).to_string());
    let mut got: Vec<C32> = push_all(&mut node, &input[..3000], 3);
    node.set_param("n", "mu_dd", json!(0.002)).unwrap();
    node.set_param("n", "mu_cma", json!(0.0001)).unwrap();
    got.extend(push_all::<C32>(&mut node, &input[3000..], 4));
    assert_eq!(got, want, "runtime parameters");
    assert!(node.set_param("n", "mu_dd", json!(1.5)).is_err());
    assert!(node.set_param("n", "feedback_taps", json!(3)).is_err());
    assert!(node.set_param("n", "phase_bw", json!(0.01)).is_err());
    assert!(node.set_param("n", "taps", json!(11)).is_err());
    assert!(node.set_param("n", "constellation", json!("bpsk")).is_err());
    assert!(node.set_param("n", "cma_symbols", json!(0)).is_err());
}

#[test]
fn psk_demod_equalizer_taps_adds_equalize() {
    let registry = Registry::standard();
    let factory = registry.factory("psk_demod").unwrap();
    let raw = |v: Value| v.as_object().unwrap().clone();
    for (params, kernels) in [
        (
            json!({ "order": 4, "symbol_rate_hz": 25_000, "equalizer_taps": 11 }),
            &["fir", "symbol_sync", "equalize", "costas"][..],
        ),
        (
            json!({ "order": 4, "symbol_rate_hz": 25_000, "equalizer_taps": 0 }),
            &["fir", "symbol_sync", "costas"][..],
        ),
        (
            json!({ "order": 4, "symbol_rate_hz": 25_000, "variant": "offset", "equalizer_taps": 7 }),
            &["fir", "symbol_sync", "rotate", "equalize"][..],
        ),
        (
            json!({ "order": 4, "symbol_rate_hz": 25_000, "variant": "offset" }),
            &["fir", "symbol_sync", "rotate"][..],
        ),
    ] {
        assert_eq!(
            factory.kernels_for(&raw(params.clone())),
            kernels,
            "{params}"
        );
    }
    // cost and delay: 8 per weight and symbol, (taps - 1) / 2 symbols of 8 samples
    for (order, taps, feedback) in [(2u64, 3u64, 0u64), (4, 11, 0), (8, 63, 0), (4, 15, 5)] {
        let report = validate(&single(
            "psk_demod",
            "c32",
            200_000,
            json!({ "order": order, "symbol_rate_hz": 25_000, "equalizer_taps": taps,
                    "equalizer_feedback_taps": feedback }),
        ));
        assert!(report.is_ok(), "{}", report.to_json());
        let soft = if order == 8 { 56.0 } else { 40.0 };
        assert_eq!(
            report.cost_per_input_sample,
            40.0 + soft / 8.0 + 8.0 * (taps + feedback) as f64 / 8.0
        );
        assert_eq!(
            report.nodes[0].delay_samples_in.reduced(),
            Delay::new(80 + 2 + (taps - 1) * 8, 2).unwrap().reduced()
        );
    }
    for taps in [1, 2, 10, 62] {
        let (path, code) = first_error(&single(
            "psk_demod",
            "c32",
            200_000,
            json!({ "symbol_rate_hz": 25_000, "equalizer_taps": taps }),
        ));
        assert_eq!(
            (path.as_str(), code),
            ("nodes.n.params.equalizer_taps", DiagCode::InvalidValue),
            "{taps}"
        );
    }
    assert!(
        !validate(&single(
            "psk_demod",
            "c32",
            200_000,
            json!({ "symbol_rate_hz": 25_000, "equalizer_taps": 65 })
        ))
        .is_ok()
    );
    let (path, code) = first_error(&single(
        "psk_demod",
        "c32",
        200_000,
        json!({ "symbol_rate_hz": 25_000, "equalizer_feedback_taps": 3 }),
    ));
    assert_eq!(
        (path.as_str(), code),
        (
            "nodes.n.params.equalizer_feedback_taps",
            DiagCode::InvalidValue
        )
    );
}

#[test]
fn psk_demod_with_equalizer_kind_equals_the_block() {
    let m = PskModConfig {
        order: 4,
        samples_per_symbol: 8,
        ..PskModConfig::default()
    };
    let mut rng = Pcg32::new(SEED, 5);
    let bits: Vec<f32> = (0..12_000)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect();
    let mut tx =
        PskModBlock::<VecStorage>::new(&m, Rate::hz(50_000).unwrap(), &mut VecStorage).unwrap();
    let mut sig = vec![C32::default(); bits.len() * 4];
    tx.process(&bits, &mut sig).unwrap();
    // an echo of 0.3 one symbol later and noise
    let noisy: Vec<C32> = (0..sig.len())
        .map(|k| {
            let echo = if k >= 8 {
                sig[k - 8] * 0.3
            } else {
                C32::default()
            };
            sig[k] + echo + C32::new(0.05 * rng.normal() as f32, 0.05 * rng.normal() as f32)
        })
        .collect();
    let d = PskDemodConfig {
        order: 4,
        symbol_rate_hz: 25_000.0,
        equalizer_taps: 11,
        equalizer_feedback_taps: 2,
        ..PskDemodConfig::default()
    };
    let fs = Rate::hz(200_000).unwrap();
    let mut block = PskDemodBlock::<VecStorage>::new(&d, fs, &mut VecStorage).unwrap();
    let mut want = vec![0.0f32; noisy.len()];
    let p = block.process(&noisy, &mut want).unwrap();
    assert_eq!(p.consumed, noisy.len());
    want.truncate(p.produced);
    let mut rx = build(
        &single(
            "psk_demod",
            "c32",
            200_000,
            json!({ "order": 4, "symbol_rate_hz": 25_000, "equalizer_taps": 11,
                    "equalizer_feedback_taps": 2 }),
        )
        .to_string(),
    );
    let soft: Vec<f32> = push_all(&mut rx, &noisy, 6);
    assert_eq!(soft.len(), want.len());
    assert!(
        soft.iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "psk_demod with the equalizer: pipeline differs from the block"
    );
    assert!(rx.set_param("n", "equalizer_taps", json!(15)).is_err());
}
