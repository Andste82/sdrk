//! Tests of a clocked node (`Clocked<f32, Packets>` input) in a pipeline.
//!
//! No kind of the standard registry has a clocked input yet, so the tests register a small
//! test kind: for every clock sample it outputs the next byte of the first queued packet as
//! an `f32` value, and -1.0 when no packet is queued.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::needless_pass_by_value
)]

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_blocks::{
    Block, BlockDescription, BlockTiming, Clocked, Packet, PacketFlags, Packets, ParamValue,
    PortDesc, PortType,
};
use serde_json::{Value, json};

use crate::kinds::{Factory, Figures, Kind};
use crate::model::Precision;
use crate::node::{ClockedNode, Node};
use crate::registry::{BuildOptions, PortKind, Registry};
use crate::storage::VecStorage;
use crate::validate::DiagCode;

static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "clocked_bytes",
    summary: "Test block: one packet byte per clock sample.",
    details: "Outputs the bytes of the queued packets, one per clock sample; -1 when idle.",
    inputs: &[
        PortDesc {
            name: "clock",
            types: &[PortType::F32],
            description: "The clock.",
        },
        PortDesc {
            name: "packets",
            types: &[PortType::Packets],
            description: "The packets.",
        },
    ],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "The bytes.",
    }],
    params: &[],
    required: &[],
};

#[derive(Debug, Default)]
struct ClockedBytes {
    /// Bytes of the first queued packet already sent.
    pos: usize,
    taken: usize,
}

impl Block for ClockedBytes {
    type In = Clocked<f32, Packets>;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(
        &mut self,
        (clock, packets): (&[f32], &[Packet]),
        output: &mut [f32],
    ) -> sdrk_blocks::Result<Progress> {
        let n = clock.len().min(output.len());
        let mut queue = packets.iter();
        let mut current = queue.next();
        self.taken = 0;
        for o in &mut output[..n] {
            // Skip empty packets.
            while let Some(p) = current
                && self.pos >= p.len()
            {
                self.pos = 0;
                self.taken += 1;
                current = queue.next();
            }
            *o = match current {
                Some(p) => {
                    let byte = p.data()[self.pos];
                    self.pos += 1;
                    f32::from(byte)
                }
                None => -1.0,
            };
        }
        if let Some(p) = current
            && self.pos >= p.len()
        {
            self.pos = 0;
            self.taken += 1;
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: Ratio::ONE,
            kind: RatioKind::Constant,
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
        self.pos = 0;
        self.taken = 0;
    }

    fn set_param(&mut self, _name: &str, _value: ParamValue) -> sdrk_blocks::Result<()> {
        Err(sdrk_blocks::Error::UnknownParameter)
    }

    fn param(&self, _name: &str) -> sdrk_blocks::Result<ParamValue> {
        Err(sdrk_blocks::Error::UnknownParameter)
    }

    fn min_input(&self) -> usize {
        1
    }

    fn consumed_packets(&self) -> usize {
        self.taken
    }
}

struct TestClockedKind;

impl Kind for TestClockedKind {
    type Cfg = ();

    const KIND: &'static str = "clocked_bytes";
    fn kernels() -> &'static [&'static str] {
        &[]
    }
    fn default_cfg() {}
    fn set(_cfg: &mut (), _name: &str, _value: ParamValue) -> sdrk_blocks::Result<()> {
        Err(sdrk_blocks::Error::UnknownParameter)
    }
    fn plan(
        _cfg: &(),
        _input: PortType,
        rate: Rate,
        _p: Precision,
    ) -> sdrk_blocks::Result<Figures> {
        Ok(Figures {
            rate_out: rate,
            memory_bytes: 0,
            cost: 1.0,
            taps: 0,
            delay: Delay::ZERO,
            min_input: 1,
            output: PortKind::Stream(PortType::F32),
        })
    }
    fn build(
        _cfg: &(),
        _input: PortType,
        _rate: Rate,
        _p: Precision,
        _st: &mut VecStorage,
    ) -> sdrk_blocks::Result<Box<dyn Node>> {
        Ok(Box::new(ClockedNode::new(ClockedBytes::default())))
    }
}

fn registry() -> Registry {
    let mut r = Registry::standard();
    r.insert_factory(Factory::<TestClockedKind>::boxed(&DESCRIPTION));
    r
}

fn flow(msg: Value) -> Value {
    json!({
        "sdrk": 1, "name": "clocked",
        "inputs": {
            "clock": { "type": "f32", "rate_hz": 1000 },
            "msg": msg
        },
        "nodes": { "tx": { "kind": "clocked_bytes" } },
        "edges": [
            { "from": "clock", "to": "tx" },
            { "from": "msg", "to": "tx", "port": "b" },
            { "from": "tx", "to": "out" }
        ],
        "outputs": { "out": { "type": "stream" } }
    })
}

fn packet(data: &[u8]) -> Packet {
    Packet::new(0, PacketFlags::NONE, data).unwrap()
}

#[test]
fn a_clocked_node_takes_packets_at_port_b_in_the_pace_of_its_clock() {
    let spec = flow(json!({ "type": "packets" }));
    let reg = registry();
    let report = reg.validate_json(&spec.to_string(), &BuildOptions::default());
    // A packet input read only at port `b` needs no rate and starts no conflicting clock.
    assert!(report.is_ok(), "{}", report.to_json());
    assert!(report.warnings.is_empty(), "{}", report.to_json());
    let node = report.nodes.iter().find(|n| n.name == "tx").unwrap();
    assert_eq!(node.clock_domain, "clock");
    assert_eq!(node.rate_out.num(), 1000);
    let mut p = reg
        .build_json(&spec.to_string(), &BuildOptions::default())
        .unwrap();
    let mut out = [0f32; 16];

    // The clock runs without packets: idle samples.
    p.push("clock", crate::InputData::F32(&[0.0; 3])).unwrap();
    assert_eq!(p.read_stream("out", &mut out).unwrap(), 3);
    assert_eq!(&out[..3], &[-1.0; 3]);

    // Packets wait for the clock.
    p.push_packets("msg", &[packet(&[1, 2, 3]), packet(&[]), packet(&[4])])
        .unwrap();
    assert_eq!(p.available("out").unwrap(), 0);
    p.push("clock", crate::InputData::F32(&[0.0; 2])).unwrap();
    assert_eq!(p.read_stream("out", &mut out).unwrap(), 2);
    assert_eq!(&out[..2], &[1.0, 2.0]);
    // The rest of the first packet, the empty one and the third one, then idle.
    p.push("clock", crate::InputData::F32(&[0.0; 4])).unwrap();
    assert_eq!(p.read_stream("out", &mut out).unwrap(), 4);
    assert_eq!(&out[..4], &[3.0, 4.0, -1.0, -1.0]);
    let stats = p.stats();
    assert_eq!(stats[0].samples_in, 9);
}

#[test]
fn a_rate_of_a_packet_input_read_only_at_port_b_is_ignored_with_a_warning() {
    let spec = flow(json!({ "type": "packets", "rate_hz": 1000 }));
    let report = registry().validate_json(&spec.to_string(), &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
    assert_eq!(report.warnings.len(), 1, "{}", report.to_json());
    assert_eq!(report.warnings[0].path, "inputs.msg.rate_hz");
    assert_eq!(report.warnings[0].code, DiagCode::InvalidValue);
    // Port `b` carries no rate or clock: another rate is no `rate_mismatch` either.
    let spec = flow(json!({ "type": "packets", "rate_hz": 777 }));
    let report = registry().validate_json(&spec.to_string(), &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
}

#[test]
fn port_b_of_a_clocked_node_takes_packets_only() {
    let spec = flow(json!({ "type": "f32", "rate_hz": 1000 }));
    let report = registry().validate_json(&spec.to_string(), &BuildOptions::default());
    assert!(
        report
            .errors
            .iter()
            .any(|d| d.code == DiagCode::TypeMismatch && d.path == "edges[1].to"),
        "{}",
        report.to_json()
    );
}

#[test]
fn reset_empties_the_packets_of_a_clocked_node() {
    let spec = flow(json!({ "type": "packets" }));
    let mut p = registry()
        .build_json(&spec.to_string(), &BuildOptions::default())
        .unwrap();
    let packets = [packet(b"ab"), packet(b"c")];
    let mut out = [0f32; 8];
    p.push_packets("msg", &packets).unwrap();
    p.push("clock", crate::InputData::F32(&[0.0; 1])).unwrap();
    assert_eq!(p.read_stream("out", &mut out).unwrap(), 1);
    p.reset();
    p.push("clock", crate::InputData::F32(&[0.0; 2])).unwrap();
    assert_eq!(p.read_stream("out", &mut out).unwrap(), 2);
    // `reset` empties the packet buffer too.
    assert_eq!(&out[..2], &[-1.0, -1.0]);
}
