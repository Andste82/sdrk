//! The `packets_in` flag of a node: set only for a node with packets in and packets out
//! (`sensor_decode`), not for an encoder (packets in, samples out), whose indices are mapped.
#![allow(clippy::unwrap_used)]

use serde_json::{Value, json};

use crate::{BuildOptions, Registry};

fn flag(kind: &str, out: &str) -> bool {
    let spec: Value = json!({
        "sdrk": 1, "name": "t", "settings": { "precision": "f32", "max_chunk": 1024 },
        "inputs": { "in": { "type": "packets", "rate_hz": 8000 } },
        "nodes": { "n": { "kind": kind, "params": {} } },
        "edges": [ { "from": "in", "to": "n" }, { "from": "n", "to": "out" } ],
        "outputs": { "out": { "type": out } }
    });
    let spec = serde_json::from_value(spec).unwrap();
    let p = Registry::standard()
        .build(&spec, &BuildOptions::default())
        .unwrap();
    p.nodes[0].packets_in
}

#[test]
fn only_packet_to_packet_nodes_skip_the_index_map() {
    assert!(flag("sensor_decode", "packets"));
    assert!(!flag("morse_encode", "stream"));
}
