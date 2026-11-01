//! The kind `preemph_limit` and the flows `fm_transmitter` and `nfm_transmitter`: registration,
//! the cost, the figures, real-only input, the parameter checks (the 40 kHz rule), the run-time
//! parameter `limit`, push mode equal to pull mode, `f32` and Q15 builds, the YAML twins, chunk
//! invariance and the clean validation.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::float_cmp,
    clippy::needless_pass_by_value
)]

mod common;

use sdrk::{Delay, Q15, Rate};
use sdrk_pipeline::{BuildOptions, InputData, Pipeline, PullError, PullSource, Registry};
use serde_json::{Value, json};

use common::{build, flow};

/// A one-node flow on a real input of type `ty` at `rate` Hz.
fn single(ty: &str, precision: &str, rate: u64, params: Value) -> Value {
    json!({
        "sdrk": 1, "name": "t", "settings": { "precision": precision, "max_chunk": 1024 },
        "inputs": { "in": { "type": ty, "rate_hz": rate } },
        "nodes": { "n": { "kind": "preemph_limit", "params": params } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": "stream" } }
    })
}

fn validate(doc: &Value) -> sdrk_pipeline::Report {
    Registry::standard().validate_json(&doc.to_string(), &BuildOptions::default())
}

#[test]
fn the_kind_is_registered_with_its_kernel() {
    let registry = Registry::standard();
    let f = registry.factory("preemph_limit").unwrap();
    assert_eq!(f.kernels(), ["preemph_limit"]);
    assert_eq!(f.description().kind, "preemph_limit");
}

#[test]
fn the_plan_has_cost_6_no_delay_and_the_rate_of_the_input() {
    for (ty, precision) in [("f32", "f32"), ("q15", "q15")] {
        for rate in [48_000u64, 240_000] {
            let report = validate(&single(ty, precision, rate, json!({})));
            assert!(report.is_ok(), "{ty} {rate}: {}", report.to_json());
            assert_eq!(report.cost_per_input_sample, 6.0);
            let node = &report.nodes[0];
            assert_eq!(node.rate_in, Rate::hz(rate).unwrap());
            assert_eq!(node.rate_out, Rate::hz(rate).unwrap());
            assert_eq!(node.delay_samples_in.reduced(), Delay::ZERO.reduced());
        }
    }
}

#[test]
fn the_input_must_be_real_and_the_parameters_are_checked() {
    assert!(!validate(&single("c32", "f32", 240_000, json!({}))).is_ok());
    assert!(!validate(&single("cq15", "q15", 240_000, json!({}))).is_ok());
    // A pre-emphasis below 40 kHz is rejected; without it any rate works.
    assert!(!validate(&single("f32", "f32", 32_000, json!({}))).is_ok());
    assert!(
        validate(&single(
            "f32",
            "f32",
            32_000,
            json!({ "preemphasis_us": 0 })
        ))
        .is_ok()
    );
    for bad in [
        json!({ "preemphasis_us": 1001 }),
        json!({ "preemphasis_us": -1 }),
        json!({ "limit": 0.0 }),
        json!({ "limit": 1.1 }),
        json!({ "hold_s": 2 }),
        json!({ "release_s": 0 }),
        json!({ "release_s": 11 }),
        json!({ "nope": 1 }),
    ] {
        let report = validate(&single("f32", "f32", 240_000, bad.clone()));
        assert!(!report.is_ok(), "{bad} must be rejected");
    }
}

#[test]
fn the_limit_can_be_changed_while_running_and_the_rest_cannot() {
    for (ty, precision) in [("f32", "f32"), ("q15", "q15")] {
        let mut p =
            build(&single(ty, precision, 240_000, json!({ "preemphasis_us": 0 })).to_string());
        assert!(p.set_param("n", "limit", json!(0.5)).is_ok());
        assert!(p.set_param("n", "limit", json!(0.05)).is_err());
        assert!(p.set_param("n", "hold_s", json!(0.1)).is_err());
        assert!(p.set_param("n", "preemphasis_us", json!(75)).is_err());
        let loud = vec![0.9f32; 64];
        let mut got = Vec::new();
        if precision == "f32" {
            p.push("in", InputData::F32(&loud)).unwrap();
            let mut buf = vec![0f32; 64];
            let n = p.read_stream::<f32>("out", &mut buf).unwrap();
            got.extend(buf[..n].iter().map(|v| f64::from(*v)));
        } else {
            let q: Vec<Q15> = loud.iter().map(|_| Q15(29491)).collect();
            p.push("in", InputData::Q15(&q)).unwrap();
            let mut buf = vec![Q15(0); 64];
            let n = p.read_stream::<Q15>("out", &mut buf).unwrap();
            got.extend(buf[..n].iter().map(|v| f64::from(v.0) / 32768.0));
        }
        assert_eq!(got.len(), 64);
        assert!(got.iter().all(|v| *v <= 0.5), "{precision}: {got:?}");
    }
}

fn audio(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.8 * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 48_000.0).sin()) as f32)
        .collect()
}

struct Source {
    data: Vec<f32>,
    pos: usize,
}

impl PullSource for Source {
    fn provide(&mut self, _input: &str, max_samples: usize) -> Result<InputData<'_>, PullError> {
        let n = max_samples.min(self.data.len() - self.pos);
        let from = self.pos;
        self.pos += n;
        Ok(InputData::F32(&self.data[from..from + n]))
    }
}

fn push_run(name: &str, data: &[f32], chunk: usize) -> Vec<u8> {
    let mut p = build(&flow(name));
    let mut out = Vec::new();
    let mut buf = vec![0u8; 8192];
    for piece in data.chunks(chunk) {
        p.push("audio", InputData::F32(piece)).unwrap();
        loop {
            let n = p.read_raw("rf", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
        }
    }
    out
}

fn pull_run(name: &str, data: Vec<f32>, want: usize) -> Vec<u8> {
    let mut p: Pipeline = build(&flow(name));
    let mut source = Source { data, pos: 0 };
    let mut buf = vec![0u8; want];
    let (n, _) = p.pull("rf", &mut buf, &mut source).unwrap();
    assert_eq!(n, want);
    buf
}

#[test]
fn both_transmitters_validate_without_warnings() {
    for name in ["fm_transmitter", "nfm_transmitter"] {
        let report = Registry::standard().validate_json(&flow(name), &BuildOptions::default());
        assert!(report.is_ok(), "{name}: {}", report.to_json());
        assert!(report.warnings.is_empty(), "{name}: {}", report.to_json());
        assert!(report.cost_per_input_sample > 0.0);
    }
}

#[test]
fn push_mode_equals_pull_mode_and_is_chunk_invariant() {
    for name in ["fm_transmitter", "nfm_transmitter"] {
        let data = audio(6000);
        let pushed = push_run(name, &data, 1000);
        assert!(pushed.len() > 400_000, "{name}: {}", pushed.len());
        // Another push size gives the same bytes.
        assert_eq!(
            push_run(name, &data, 777)[..300_000],
            pushed[..300_000],
            "{name}"
        );
        // Pull mode gives the same prefix.
        let pulled = pull_run(name, data, 300_000);
        assert_eq!(pulled[..], pushed[..300_000], "{name}");
        // The signal is not silence.
        assert!(pushed.iter().any(|b| *b != 0));
    }
}

#[test]
fn the_flows_run_in_q15() {
    for name in ["fm_transmitter", "nfm_transmitter"] {
        let mut v: Value = serde_json::from_str(&flow(name)).unwrap();
        v["settings"]["precision"] = json!("q15");
        v["inputs"]["audio"]["type"] = json!("q15");
        let mut p = build(&v.to_string());
        let q: Vec<Q15> = audio(4800)
            .iter()
            .map(|x| Q15((f64::from(*x) * 32768.0) as i16))
            .collect();
        p.push("audio", InputData::Q15(&q)).unwrap();
        let mut buf = vec![0u8; 1 << 16];
        let mut total = 0;
        loop {
            let n = p.read_raw("rf", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            total += n;
        }
        assert!(total > 400_000, "{name}: {total}");
    }
}

#[test]
fn the_shape_node_of_the_transmitters_is_a_preemph_limit() {
    let v: Value = serde_json::from_str(&flow("fm_transmitter")).unwrap();
    assert_eq!(v["nodes"]["shape"]["kind"], "preemph_limit");
    assert_eq!(v["nodes"]["shape"]["params"]["preemphasis_us"], 50);
    assert_eq!(v["nodes"]["shape"]["params"]["limit"], 1.0);
    let v: Value = serde_json::from_str(&flow("nfm_transmitter")).unwrap();
    assert_eq!(v["nodes"]["shape"]["params"]["preemphasis_us"], 530);
    assert_eq!(v["nodes"]["shape"]["params"]["limit"], 0.85);
    assert_eq!(v["nodes"]["mod"]["params"]["deviation_hz"], 5000);
    // The limit of the running flow can be changed.
    let mut p = build(&flow("nfm_transmitter"));
    p.set_param("shape", "limit", json!(0.5)).unwrap();
}
