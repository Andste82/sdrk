//! Property tests: random graphs from the registry, random bytes and random JSON.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::assert_is_empty,
    clippy::unreadable_literal,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::redundant_closure_for_method_calls
)]

mod common;

use proptest::prelude::*;
use sdrk::{C32, CQ15, Q15};
use sdrk_pipeline::{BuildOptions, InputData, PipelineSpec, Registry};
use sdrk_testkit::Pcg32;
use serde_json::{Map, Value, json};

const RATES: [u64; 5] = [8_000, 48_000, 240_000, 1_000_000, 2_400_000];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Ty {
    Raw,
    Complex,
    Real,
    Frames,
}

fn pick<'a, T>(rng: &mut Pcg32, items: &'a [T]) -> &'a T {
    &items[rng.below(items.len() as u32) as usize]
}

fn chance(rng: &mut Pcg32, percent: u32) -> bool {
    rng.below(100) < percent
}

/// Parameters of a node of `kind` at the input rate `rate`; mostly valid, sometimes not.
fn params(rng: &mut Pcg32, kind: &str, rate: f64) -> Value {
    let sloppy = chance(rng, 12);
    let frac = |rng: &mut Pcg32, lo: f64, hi: f64| lo + (hi - lo) * rng.next_f64();
    let mut p = Map::new();
    match kind {
        "convert" | "convert_out" => {
            p.insert(
                "format".into(),
                json!(*pick(
                    rng,
                    &[
                        "cu8",
                        "cs8",
                        "cs16",
                        "cs12_lsb16",
                        "sc16_q11",
                        "cf32",
                        "packed12"
                    ]
                )),
            );
        }
        "scale" => {
            p.insert(
                "gain_db".into(),
                json!(frac(rng, -40.0, if sloppy { 60.0 } else { 0.0 })),
            );
        }
        "dc_block" => {
            p.insert(
                "cutoff_hz".into(),
                json!(frac(rng, 1.0, rate * if sloppy { 0.9 } else { 0.1 })),
            );
        }
        "fir" => {
            p.insert("cutoff_hz".into(), json!(rate * frac(rng, 0.05, 0.2)));
            p.insert("transition_hz".into(), json!(rate * frac(rng, 0.03, 0.1)));
            if sloppy {
                p.insert("attenuation_db".into(), json!(frac(rng, 0.0, 300.0)));
            }
        }
        "ddc" => {
            p.insert(
                "decimation".into(),
                json!(1 + rng.below(if sloppy { 3000 } else { 20 })),
            );
            p.insert(
                "frequency_hz".into(),
                json!(rate * frac(rng, -0.4, if sloppy { 0.9 } else { 0.4 })),
            );
            p.insert("max_chunk".into(), json!(64 + rng.below(1000)));
        }
        "duc" => {
            p.insert(
                "interpolation".into(),
                json!(1 + rng.below(if sloppy { 3000 } else { 6 })),
            );
            p.insert("frequency_hz".into(), json!(rate * frac(rng, -0.4, 0.4)));
        }
        "fm_demod" => {
            p.insert(
                "audio_decimation".into(),
                json!(1 + rng.below(if sloppy { 100 } else { 4 })),
            );
            p.insert(
                "audio_cutoff_hz".into(),
                json!(rate * frac(rng, 0.01, 0.08)),
            );
            p.insert("deemphasis_us".into(), json!(frac(rng, 0.0, 100.0)));
        }
        "fm_mod" => {
            p.insert(
                "deviation_hz".into(),
                json!(rate * frac(rng, 0.0, if sloppy { 2.0 } else { 0.4 })),
            );
        }
        "resampler" => {
            p.insert(
                "rate_out_hz".into(),
                json!(*pick(rng, &[8000u64, 16000, 24000, 48000, 96000])),
            );
            p.insert("tunable".into(), json!(chance(rng, 50)));
            p.insert("max_taps".into(), json!(4096));
        }
        "spectrum" => {
            p.insert(
                "fft_size".into(),
                json!(1u32 << (4 + rng.below(if sloppy { 14 } else { 6 }))),
            );
            p.insert("averaging".into(), json!(1 + rng.below(3)));
            if chance(rng, 30) {
                p.insert(
                    "window".into(),
                    json!(*pick(rng, &["hann", "blackman_harris", "kaiser"])),
                );
            }
        }
        _ => {}
    }
    Value::Object(p)
}

/// A random flow: type-aware most of the time, with a few random mistakes mixed in.
fn random_flow(seed: u64) -> Value {
    let mut rng = Pcg32::new(seed, 7);
    let q15 = chance(&mut rng, 40);
    let mut inputs = Map::new();
    // (name, type, rate)
    let mut streams: Vec<(String, Ty, f64)> = Vec::new();
    for i in 0..=rng.below(2) {
        let name = format!("in{i}");
        let rate = *pick(&mut rng, &RATES);
        let (spec, ty) = match rng.below(3) {
            0 => (
                json!({ "format": *pick(&mut rng, &["cu8", "cs8", "cs16", "cf32"]), "rate_hz": rate }),
                Ty::Raw,
            ),
            1 => (
                json!({ "type": if q15 { "cq15" } else { "c32" }, "rate_hz": rate }),
                Ty::Complex,
            ),
            _ => (
                json!({ "type": if q15 { "q15" } else { "f32" }, "rate_hz": rate }),
                Ty::Real,
            ),
        };
        inputs.insert(name.clone(), spec);
        // A raw input is converted to complex.
        streams.push((
            name,
            if ty == Ty::Raw { Ty::Complex } else { ty },
            rate as f64,
        ));
    }
    let kinds_complex = [
        "scale",
        "dc_block",
        "fir",
        "ddc",
        "duc",
        "fm_demod",
        "resampler",
        "spectrum",
        "convert_out",
    ];
    let kinds_real = ["resampler", "fm_mod"];
    let mut nodes = Map::new();
    let mut edges = Vec::new();
    let mut used = vec![false; 0];
    let mut outputs = Map::new();
    let count = 1 + rng.below(6);
    for i in 0..count {
        let usable: Vec<usize> = streams
            .iter()
            .enumerate()
            .filter(|(_, s)| s.1 != Ty::Frames)
            .map(|(k, _)| k)
            .collect();
        if usable.is_empty() {
            break;
        }
        let parent = *pick(&mut rng, &usable);
        let (pname, pty, prate) = streams[parent].clone();
        let kind = match pty {
            Ty::Complex => *pick(&mut rng, &kinds_complex),
            Ty::Real => *pick(&mut rng, &kinds_real),
            Ty::Raw => "convert",
            Ty::Frames => unreachable!(),
        };
        let name = format!("n{i}");
        nodes.insert(
            name.clone(),
            json!({ "kind": kind, "params": params(&mut rng, kind, prate) }),
        );
        edges.push(json!({ "from": pname, "to": name }));
        let rate_out = match kind {
            "ddc" => prate / nodes[&name]["params"]["decimation"].as_f64().unwrap_or(1.0),
            "duc" => {
                prate
                    * nodes[&name]["params"]["interpolation"]
                        .as_f64()
                        .unwrap_or(1.0)
            }
            "fm_demod" => {
                prate
                    / nodes[&name]["params"]["audio_decimation"]
                        .as_f64()
                        .unwrap_or(1.0)
            }
            "resampler" => nodes[&name]["params"]["rate_out_hz"]
                .as_f64()
                .unwrap_or(prate),
            _ => prate,
        };
        let out_ty = match kind {
            "fm_demod" => Ty::Real,
            "fm_mod" => Ty::Complex,
            "spectrum" => Ty::Frames,
            "convert_out" => Ty::Raw,
            _ => pty,
        };
        streams.push((name, out_ty, rate_out));
        used.push(false);
    }
    // Every stream that nobody reads gets an output (most of the time).
    let read: Vec<String> = edges
        .iter()
        .map(|e| e["from"].as_str().unwrap().to_string())
        .collect();
    for (name, ty, _) in &streams {
        if read.contains(name) && !chance(&mut rng, 3) {
            continue;
        }
        if chance(&mut rng, 4) {
            continue;
        }
        let out = format!("o_{name}");
        let spec = match ty {
            Ty::Frames => {
                json!({ "type": "frames", "queue": 1 + rng.below(4), "drop": *pick(&mut rng, &["error", "drop_oldest", "drop_newest"]) })
            }
            _ => {
                json!({ "type": "stream", "capacity": 64 + rng.below(8000), "drop": *pick(&mut rng, &["error", "drop_oldest", "drop_newest"]) })
            }
        };
        outputs.insert(out.clone(), spec);
        edges.push(json!({ "from": name, "to": out }));
    }
    // A few random mistakes.
    if chance(&mut rng, 10) {
        edges.push(
            json!({ "from": format!("n{}", rng.below(8)), "to": format!("n{}", rng.below(8)) }),
        );
    }
    if chance(&mut rng, 5) && !edges.is_empty() {
        let k = rng.below(edges.len() as u32) as usize;
        edges.remove(k);
    }
    json!({
        "sdrk": 1, "name": "random",
        "settings": { "precision": if q15 { "q15" } else { "f32" }, "max_chunk": 64 + rng.below(2000) },
        "inputs": inputs, "nodes": nodes, "edges": edges, "outputs": outputs
    })
}

fn noise_float(rng: &mut Pcg32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| (rng.next_f64() * 2.0 - 1.0) as f32)
        .collect()
}

fn push_noise(p: &mut sdrk_pipeline::Pipeline, spec: &PipelineSpec, rng: &mut Pcg32) {
    let max = spec.settings.max_chunk;
    for (name, input) in &spec.inputs {
        for _ in 0..3 {
            let n = 1 + rng.below(max as u32) as usize;
            let result = if input.format.is_some() {
                let bytes: Vec<u8> = (0..n.min(max)).map(|_| rng.next_u32() as u8).collect();
                p.push(name, InputData::Raw(&bytes))
            } else {
                match input.sample_type.unwrap() {
                    sdrk_pipeline::InputType::C32 => {
                        let v: Vec<C32> = noise_float(rng, 2 * n)
                            .chunks(2)
                            .map(|c| C32::new(c[0], c[1]))
                            .collect();
                        p.push(name, InputData::C32(&v))
                    }
                    sdrk_pipeline::InputType::Cq15 => {
                        let v: Vec<CQ15> = (0..n)
                            .map(|_| CQ15 {
                                re: Q15(rng.next_u32() as i16),
                                im: Q15(rng.next_u32() as i16),
                            })
                            .collect();
                        p.push(name, InputData::CQ15(&v))
                    }
                    sdrk_pipeline::InputType::F32 => {
                        p.push(name, InputData::F32(&noise_float(rng, n)))
                    }
                    sdrk_pipeline::InputType::Q15 => {
                        let v: Vec<Q15> = (0..n).map(|_| Q15(rng.next_u32() as i16)).collect();
                        p.push(name, InputData::Q15(&v))
                    }
                }
            };
            result.unwrap_or_else(|e| panic!("push to {name} failed: {e}"));
            // Read everything so that no `error` output stays full.
            for oname in spec.outputs.keys() {
                drain(p, oname);
            }
        }
    }
}

fn drain(p: &mut sdrk_pipeline::Pipeline, oname: &str) {
    let mut frame = vec![0f32; 1 << 17];
    let mut bytes = vec![0u8; 1 << 12];
    loop {
        let mut moved = false;
        // Whichever kind the output is, one of the readers accepts it.
        if let Ok(n) = p.read_raw(oname, &mut bytes) {
            moved |= n > 0;
        } else if let Ok(n) = p.read_stream(oname, &mut [C32::default(); 512]) {
            moved |= n > 0;
        } else if let Ok(n) = p.read_stream(oname, &mut [CQ15::default(); 512]) {
            moved |= n > 0;
        } else if let Ok(n) = p.read_stream(oname, &mut [0f32; 512]) {
            moved |= n > 0;
        } else if let Ok(n) = p.read_stream(oname, &mut [Q15(0); 512]) {
            moved |= n > 0;
        } else {
            let len = p
                .available(oname)
                .map(|_| ())
                .ok()
                .and_then(|()| frame_len(p, oname));
            if let Some(len) = len {
                if let Ok(Some(_)) = p.read_frame(oname, &mut frame[..len]) {
                    moved = true;
                }
            }
        }
        if !moved {
            return;
        }
    }
}

fn frame_len(p: &mut sdrk_pipeline::Pipeline, oname: &str) -> Option<usize> {
    match p.read_frame(oname, &mut []) {
        Err(sdrk_pipeline::Error::LengthMismatch { expected, .. }) => Some(expected),
        _ => None,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Random graphs: validation never panics; whatever validates also builds and runs.
    #[test]
    fn random_graphs_validate_build_and_run(seed in any::<u64>()) {
        let registry = Registry::standard();
        let opts = BuildOptions::default();
        let flow = random_flow(seed);
        let text = flow.to_string();
        let report = registry.validate_json(&text, &opts);
        // The report is always serializable.
        let _ = report.to_json();
        if report.is_ok() {
            let spec = PipelineSpec::from_json(&text).unwrap();
            let mut pipeline = registry.build(&spec, &opts)
                .unwrap_or_else(|r| panic!("validated but did not build: {}\n{text}", r.to_json()));
            let mut rng = Pcg32::new(seed, 99);
            push_noise(&mut pipeline, &spec, &mut rng);
            // The readout is a valid description of the same pipeline.
            prop_assert_eq!(pipeline.config(), spec);
        } else {
            let spec = serde_json::from_value::<PipelineSpec>(flow);
            if let Ok(spec) = spec {
                prop_assert!(registry.build(&spec, &opts).is_err());
            }
        }
    }

    /// Random byte strings never panic.
    #[test]
    fn random_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..300)) {
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let registry = Registry::standard();
        let _ = PipelineSpec::from_json(&text);
        let report = registry.validate_json(&text, &BuildOptions::default());
        prop_assert!(!report.is_ok() || text.contains("sdrk"));
        let _ = report.to_json();
    }

    /// Random JSON values never panic.
    #[test]
    fn random_json_values_never_panic(value in arb_json()) {
        let text = value.to_string();
        let registry = Registry::standard();
        let _ = PipelineSpec::from_json(&text);
        let report = registry.validate_json(&text, &BuildOptions::default());
        prop_assert!(!report.is_ok());
    }

    /// A valid flow with a few corrupted bytes is rejected or accepted, never a panic.
    #[test]
    fn corrupted_flows_never_panic(index in 0usize..4000, byte in any::<u8>(), count in 1usize..4) {
        let registry = Registry::standard();
        let mut bytes = common::flow("fm_receiver").into_bytes();
        for k in 0..count {
            let at = (index + 37 * k) % bytes.len();
            bytes[at] = byte.wrapping_add(k as u8);
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let opts = BuildOptions::default();
        let report = registry.validate_json(&text, &opts);
        if report.is_ok() {
            prop_assert!(registry.build_json(&text, &opts).is_ok());
        }
    }

    /// A random object with the right keys but random content.
    #[test]
    fn random_values_in_the_right_places_never_panic(
        a in arb_json(), b in arb_json(), c in arb_json(), d in arb_json()
    ) {
        let doc = json!({
            "sdrk": 1, "name": "x",
            "settings": a,
            "inputs": { "i": b },
            "nodes": { "n": c },
            "edges": [ { "from": "i", "to": "n" }, d ],
            "outputs": {}
        });
        let report = Registry::standard().validate_json(&doc.to_string(), &BuildOptions::default());
        prop_assert!(!report.is_ok());
    }
}

fn arb_json() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(|n| json!(n)),
        any::<f64>().prop_map(|f| json!(f)),
        "[ -~]{0,12}".prop_map(Value::String),
    ];
    leaf.prop_recursive(4, 48, 6, |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 0..5).prop_map(Value::Array),
            proptest::collection::btree_map("[a-z_]{0,10}", inner, 0..5)
                .prop_map(|m| Value::Object(m.into_iter().collect())),
        ]
    })
}

#[test]
fn the_generator_produces_valid_flows_often_enough() {
    // Guards the property test against a generator that only makes invalid graphs.
    let registry = Registry::standard();
    let valid = (0..300u64)
        .filter(|seed| {
            registry
                .validate_json(&random_flow(*seed).to_string(), &BuildOptions::default())
                .is_ok()
        })
        .count();
    assert!(valid >= 60, "only {valid} of 300 random flows are valid");
}
