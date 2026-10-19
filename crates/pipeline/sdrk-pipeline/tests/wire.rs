//! The wire format: round trips, forward compatibility, errors and the committed bytes.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::assert_is_empty,
    clippy::unreadable_literal,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::doc_markdown,
    clippy::cast_possible_truncation
)]

mod common;

use proptest::prelude::*;
use sdrk::{C32, CQ15, Q15};
use sdrk_blocks::{FrameMeta, ParamValue};
use sdrk_device::{BlockFlags, SampleIndex, TimeAnchor, TimeSource};
use sdrk_device::{ErrorKind, SdrkError};
use sdrk_pipeline::wire::{
    Decoded, HEADER_LEN, Hello, MAGIC, Message, MessageType, OwnedData, PayloadType, VERSION,
    WireAnchor, WireEvent, WireEventKind, WireParamAck, WireParamSet, WireParamTime, decode,
    decode_stream, encode,
};
use sdrk_pipeline::{Error, Event, EventKind, NodeId, OutputId, ParamTime};
use serde_json::json;

use common::workspace_file;

fn bytes_of(m: &Message) -> Vec<u8> {
    let mut out = Vec::new();
    encode(m, &mut out);
    out
}

fn committed(name: &str) -> Vec<u8> {
    std::fs::read(workspace_file(&format!(
        "crates/pipeline/sdrk-pipeline/tests/wire/{name}.bin"
    )))
    .unwrap()
}

/// The six example messages of `docs/wire-format.md`.
fn examples() -> Vec<(&'static str, Message)> {
    let anchorless = Event {
        output: OutputId::new_for_test(0),
        index: 12345,
        kind: EventKind::Gap { missing: 7 },
    };
    vec![
        (
            "samples",
            Message::samples_c32(
                3,
                1000,
                BlockFlags::OVERFLOW,
                &[C32::new(1.0, -2.0), C32::new(0.5, 0.25)],
            ),
        ),
        (
            "frame",
            Message::frame(
                5,
                &FrameMeta {
                    first_sample: 4096,
                    averaged: 8,
                    exponent: 0,
                },
                &[0.0, -10.5, -20.25, 3.0],
            ),
        ),
        ("event", Message::event(2, &anchorless)),
        (
            "hello",
            Message::hello(&Hello::samples(1, "audio", "f32", 48000, 1)),
        ),
        (
            "param_set",
            Message::param_set(
                0,
                &WireParamSet {
                    node: "tune".into(),
                    name: "frequency_hz".into(),
                    value: json!(610_000.0),
                    at: WireParamTime::InputIndex {
                        input: "rf".into(),
                        index: 100,
                    },
                },
            ),
        ),
        (
            "param_ack",
            Message::param_ack(
                0,
                &WireParamAck {
                    node: "tune".into(),
                    name: "frequency_hz".into(),
                    index: 100,
                    late: false,
                },
            ),
        ),
    ]
}

trait OutputIdForTest {
    fn new_for_test(i: usize) -> Self;
}

impl OutputIdForTest for OutputId {
    fn new_for_test(_i: usize) -> Self {
        // Ids come from a pipeline; the first output of a pipeline has id 0.
        let mut p = common::build(&common::flow("fm_receiver"));
        let id = p.output_id("audio").unwrap();
        let _ = &mut p;
        id
    }
}

#[test]
fn the_header_is_32_bytes_little_endian() {
    assert_eq!(HEADER_LEN, 32);
    assert_eq!(&MAGIC, b"SDRK");
    assert_eq!(VERSION, 1);
    let m = Message {
        kind: MessageType::Frame,
        flags: BlockFlags::LATE.union(BlockFlags::RETUNED),
        stream: 0x0102,
        payload_type: PayloadType::F32,
        start: 0x0807_0605_0403_0201,
        count: 1,
        averaged: 0x0D0C_0B0A,
        payload: vec![0, 0, 0x80, 0x3f],
    };
    let b = bytes_of(&m);
    assert_eq!(b.len(), 36);
    assert_eq!(&b[0..4], b"SDRK");
    assert_eq!(b[4], 1);
    assert_eq!(b[5], 2);
    assert_eq!(u16::from_le_bytes([b[6], b[7]]), (1 << 5) | (1 << 6));
    assert_eq!(&b[8..10], &[0x02, 0x01]);
    assert_eq!(b[10], 3);
    assert_eq!(b[11], 0);
    assert_eq!(&b[12..20], &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(&b[20..24], &[1, 0, 0, 0]);
    assert_eq!(&b[24..28], &[4, 0, 0, 0]);
    assert_eq!(&b[28..32], &[0x0A, 0x0B, 0x0C, 0x0D]);
    assert_eq!(&b[32..], &[0, 0, 0x80, 0x3f]);
    // Only frames carry `averaged`.
    let samples = Message {
        kind: MessageType::Samples,
        ..m
    };
    assert_eq!(&bytes_of(&samples)[28..32], &[0, 0, 0, 0]);
}

#[test]
fn the_codes_are_those_of_the_spec() {
    let kinds = [
        (MessageType::Samples, 1),
        (MessageType::Frame, 2),
        (MessageType::Event, 3),
        (MessageType::Hello, 4),
        (MessageType::ParamSet, 5),
        (MessageType::ParamAck, 6),
    ];
    for (k, code) in kinds {
        assert_eq!(k.code(), code);
        assert_eq!(MessageType::from_code(code), Some(k));
    }
    let payloads = [
        (PayloadType::Raw, 0),
        (PayloadType::C32, 1),
        (PayloadType::Cq15, 2),
        (PayloadType::F32, 3),
        (PayloadType::Q15, 4),
        (PayloadType::Json, 5),
    ];
    for (k, code) in payloads {
        assert_eq!(k.code(), code);
        assert_eq!(PayloadType::from_code(code), Some(k));
    }
    assert_eq!(MessageType::from_code(0), None);
    assert_eq!(MessageType::from_code(7), None);
    assert_eq!(PayloadType::from_code(6), None);
}

#[test]
fn every_message_type_round_trips() {
    for (name, m) in examples() {
        let bytes = bytes_of(&m);
        let Decoded::Message(back, used) = decode(&bytes).unwrap() else {
            panic!("{name}")
        };
        assert_eq!(back, m, "{name}");
        assert_eq!(used, bytes.len(), "{name}");
    }
    // The typed accessors.
    let (_, samples) = &examples()[0];
    assert_eq!(
        samples.to_data().unwrap(),
        OwnedData::C32(vec![C32::new(1.0, -2.0), C32::new(0.5, 0.25)])
    );
    let (_, frame) = &examples()[1];
    let (meta, values) = frame.to_frame().unwrap();
    assert_eq!((meta.first_sample, meta.averaged), (4096, 8));
    assert_eq!(values, [0.0, -10.5, -20.25, 3.0]);
    let (_, hello) = &examples()[3];
    assert_eq!(hello.to_hello().unwrap().name, "audio");
    let (_, set) = &examples()[4];
    assert_eq!(set.to_param_set().unwrap().node, "tune");
    let (_, ack) = &examples()[5];
    assert_eq!(ack.to_param_ack().unwrap().index, 100);
    // The wrong accessor is an error.
    assert!(matches!(hello.to_data(), Err(Error::Wire(_))));
    assert!(matches!(samples.to_hello(), Err(Error::Wire(_))));
    assert!(matches!(samples.to_frame(), Err(Error::Wire(_))));
}

#[test]
fn every_sample_type_round_trips() {
    let msgs = [
        Message::samples_raw(1, 5, BlockFlags::empty(), 3, &[1, 2, 3, 4, 5, 6]),
        Message::samples_c32(1, 5, BlockFlags::empty(), &[C32::new(1.0, 2.0)]),
        Message::samples_cq15(
            1,
            5,
            BlockFlags::GAP_BEFORE,
            &[CQ15 {
                re: Q15(1),
                im: Q15(-2),
            }],
        ),
        Message::samples_f32(1, 5, BlockFlags::empty(), &[0.5, f32::MAX]),
        Message::samples_q15(1, 5, BlockFlags::empty(), &[Q15(i16::MIN), Q15(i16::MAX)]),
    ];
    for m in msgs {
        let bytes = bytes_of(&m);
        let Decoded::Message(back, _) = decode(&bytes).unwrap() else {
            panic!()
        };
        assert_eq!(back, m);
        let data = back.to_data().unwrap();
        let again = Message::from_input(1, 5, back.flags, m.count as usize, &data.as_input());
        assert_eq!(again, m);
    }
}

#[test]
fn the_committed_bytes_are_the_examples() {
    for (name, m) in examples() {
        let file = committed(match name {
            "param_set" => "param_set",
            "param_ack" => "param_ack",
            other => other,
        });
        assert_eq!(bytes_of(&m), file, "{name}: encoder");
        let Decoded::Message(back, used) = decode(&file).unwrap() else {
            panic!("{name}")
        };
        assert_eq!(back, m, "{name}: decoder");
        assert_eq!(used, file.len());
    }
    // The size of each example is documented in docs/wire-format.md.
    assert_eq!(committed("samples").len(), 48);
    assert_eq!(committed("frame").len(), 48);
}

#[test]
fn truncated_buffers_need_more() {
    for (name, m) in examples() {
        let bytes = bytes_of(&m);
        for cut in 0..bytes.len() {
            match decode(&bytes[..cut]).unwrap() {
                Decoded::NeedMore(n) => {
                    assert!(n >= 1, "{name} at {cut}");
                    // Exactly the missing bytes once the header is complete.
                    if cut >= HEADER_LEN {
                        assert_eq!(n, bytes.len() - cut, "{name} at {cut}");
                    }
                    // Supplying `n` more bytes never decodes less than the whole message.
                    assert!(cut + n <= bytes.len());
                }
                other => panic!("{name} at {cut}: {other:?}"),
            }
        }
    }
    assert_eq!(decode(&[]).unwrap(), Decoded::NeedMore(HEADER_LEN));
}

#[test]
fn bad_magic_version_and_lengths_are_parse_errors() {
    let good = bytes_of(&examples()[0].1);
    let mut bad = good.clone();
    bad[0] = b'X';
    let err = decode(&bad).unwrap_err();
    assert!(matches!(err, Error::Wire(_)));
    assert_eq!(err.kind(), ErrorKind::Parse);
    // A partial wrong magic is detected at once.
    assert!(decode(b"SDX").is_err());
    assert!(decode(b"S").is_ok());
    let mut bad = good.clone();
    bad[4] = 2;
    assert!(decode(&bad).is_err());
    assert!(
        decode(&bad[..6]).is_err(),
        "the version is checked before the header is complete"
    );
    // Unknown payload type of a known message.
    let mut bad = good.clone();
    bad[10] = 9;
    assert!(decode(&bad).is_err());
    // The element count does not match the payload length.
    let mut bad = good.clone();
    bad[20] = 3;
    assert!(decode(&bad).is_err());
    // A payload that is too large.
    let mut bad = good.clone();
    bad[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(decode(&bad).is_err());
    assert!(decode_stream(&bad).is_err());
}

#[test]
fn unknown_message_types_are_skipped() {
    let mut unknown = bytes_of(&examples()[0].1);
    unknown[5] = 200;
    unknown[10] = 99; // not even a payload type this version knows
    let first = bytes_of(&examples()[3].1);
    let last = bytes_of(&examples()[1].1);
    let mut stream = first.clone();
    stream.extend_from_slice(&unknown);
    stream.extend_from_slice(&last);
    assert_eq!(decode(&unknown).unwrap(), Decoded::Skipped(unknown.len()));
    assert!(matches!(
        decode(&unknown[..unknown.len() - 1]).unwrap(),
        Decoded::NeedMore(1)
    ));
    let (messages, used) = decode_stream(&stream).unwrap();
    assert_eq!(used, stream.len());
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].kind, MessageType::Hello);
    assert_eq!(messages[1].kind, MessageType::Frame);
    // An incomplete last message is left in the buffer.
    let (messages, used) = decode_stream(&stream[..stream.len() - 5]).unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(used, first.len() + unknown.len());
}

#[test]
fn events_round_trip_through_the_mirror_types() {
    let node = NodeId::new_for_test();
    let all = [
        EventKind::Gap { missing: 12 },
        EventKind::Overflow,
        EventKind::Underrun,
        EventKind::Anchor(TimeAnchor {
            index: SampleIndex(77),
            unix_ns: -5,
            source: TimeSource::Gps,
        }),
        EventKind::Retuned,
        EventKind::BurstStart {
            tx_time_ns: Some(9),
        },
        EventKind::BurstStart { tx_time_ns: None },
        EventKind::BurstEnd,
        EventKind::Late,
        EventKind::ParamApplied {
            node,
            param: 3,
            value: ParamValue::F64(0.25),
            late: true,
        },
        EventKind::ParamApplied {
            node,
            param: 0,
            value: ParamValue::Format(sdrk_device::formats::CS16),
            late: false,
        },
        EventKind::EventsLost { count: 4 },
    ];
    let output = OutputId::new_for_test(0);
    for kind in all {
        let event = Event {
            output,
            index: 1_000_000_007,
            kind,
        };
        let m = Message::event(9, &event);
        let bytes = bytes_of(&m);
        let Decoded::Message(back, _) = decode(&bytes).unwrap() else {
            panic!()
        };
        assert_eq!(back.start, 1_000_000_007);
        assert_eq!(back.to_event(output).unwrap(), event, "{kind:?}");
    }
    // The tag is snake_case.
    let text = String::from_utf8(
        Message::event(
            0,
            &Event {
                output,
                index: 0,
                kind: EventKind::BurstStart { tx_time_ns: None },
            },
        )
        .payload,
    )
    .unwrap();
    assert_eq!(
        text,
        r#"{"index":0,"kind":{"type":"burst_start","tx_time_ns":null}}"#
    );
    // Bad payloads.
    let mut broken = Message::event(
        0,
        &Event {
            output,
            index: 0,
            kind: EventKind::Overflow,
        },
    );
    broken.payload = b"{\"index\":0,\"kind\":{\"type\":\"nope\"}}".to_vec();
    assert!(matches!(broken.to_event(output), Err(Error::Wire(_))));
    let anchor = WireEvent {
        index: 0,
        kind: WireEventKind::Anchor {
            anchor: WireAnchor {
                index: 0,
                unix_ns: 0,
                source: "mars".into(),
            },
        },
    };
    assert!(anchor.into_event(output).is_err());
}

trait NodeIdForTest {
    fn new_for_test() -> Self;
}

impl NodeIdForTest for NodeId {
    fn new_for_test() -> Self {
        common::build(&common::flow("fm_receiver"))
            .node_id("tune")
            .unwrap()
    }
}

#[test]
fn parameter_messages_carry_the_time() {
    for t in [
        ParamTime::Immediate,
        ParamTime::InputIndex {
            input: "rf".into(),
            index: u64::MAX,
        },
        ParamTime::UnixNs(-1),
    ] {
        let set = WireParamSet {
            node: "n".into(),
            name: "p".into(),
            value: json!(true),
            at: WireParamTime::from(&t),
        };
        let back = Message::param_set(4, &set).to_param_set().unwrap();
        assert_eq!(back, set);
        assert_eq!(ParamTime::from(back.at), t);
    }
}

// ------------------------------------------------------------------ property test

fn arb_message() -> impl Strategy<Value = Message> {
    let flags = (0u16..0x80).prop_map(BlockFlags::from_bits_truncate);
    let json_text = "[a-z]{0,40}".prop_map(|s| format!("{{\"k\":\"{s}\"}}").into_bytes());
    let samples = (
        flags.clone(),
        any::<u16>(),
        any::<u64>(),
        prop_oneof![
            Just(PayloadType::C32),
            Just(PayloadType::Cq15),
            Just(PayloadType::F32),
            Just(PayloadType::Q15)
        ],
        0usize..40,
    )
        .prop_flat_map(|(flags, stream, start, ty, n)| {
            let size = ty.element_size().unwrap();
            proptest::collection::vec(any::<u8>(), n * size).prop_map(move |payload| Message {
                kind: MessageType::Samples,
                flags,
                stream,
                payload_type: ty,
                start,
                count: n as u32,
                averaged: 0,
                payload,
            })
        });
    let raw = (
        flags,
        any::<u16>(),
        any::<u64>(),
        0u32..100,
        proptest::collection::vec(any::<u8>(), 0..200),
    )
        .prop_map(|(flags, stream, start, count, payload)| Message {
            kind: MessageType::Samples,
            flags,
            stream,
            payload_type: PayloadType::Raw,
            start,
            count,
            averaged: 0,
            payload,
        });
    let frame = (any::<u16>(), any::<u64>(), any::<u32>(), 0usize..30).prop_flat_map(
        |(stream, start, averaged, n)| {
            proptest::collection::vec(any::<u8>(), n * 4).prop_map(move |payload| Message {
                kind: MessageType::Frame,
                flags: BlockFlags::empty(),
                stream,
                payload_type: PayloadType::F32,
                start,
                count: n as u32,
                averaged,
                payload,
            })
        },
    );
    let json_msg = (
        prop_oneof![
            Just(MessageType::Event),
            Just(MessageType::Hello),
            Just(MessageType::ParamSet),
            Just(MessageType::ParamAck)
        ],
        any::<u16>(),
        any::<u64>(),
        json_text,
    )
        .prop_map(|(kind, stream, start, payload)| Message {
            kind,
            flags: BlockFlags::empty(),
            stream,
            payload_type: PayloadType::Json,
            start,
            count: payload.len() as u32,
            averaged: 0,
            payload,
        });
    prop_oneof![samples, raw, frame, json_msg]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn random_messages_round_trip(msgs in proptest::collection::vec(arb_message(), 1..8), cut in 0usize..64) {
        let mut stream = Vec::new();
        for m in &msgs {
            let one = bytes_of(m);
            let Decoded::Message(back, used) = decode(&one).unwrap() else { panic!() };
            prop_assert_eq!(&back, m);
            prop_assert_eq!(used, one.len());
            stream.extend_from_slice(&one);
        }
        let (all, used) = decode_stream(&stream).unwrap();
        prop_assert_eq!(&all, &msgs);
        prop_assert_eq!(used, stream.len());
        // Cutting the stream anywhere decodes a prefix of the messages and never fails.
        let at = stream.len().saturating_sub(cut);
        let (part, used) = decode_stream(&stream[..at]).unwrap();
        prop_assert!(used <= at);
        prop_assert_eq!(&part[..], &msgs[..part.len()]);
    }

    #[test]
    fn random_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..200)) {
        let _ = decode(&bytes);
        let _ = decode_stream(&bytes);
    }

    #[test]
    fn corrupted_messages_never_panic(m in arb_message(), at in 0usize..300, byte in any::<u8>()) {
        let mut b = bytes_of(&m);
        let n = b.len();
        b[at % n] = byte;
        let _ = decode(&b);
        let _ = decode_stream(&b);
    }
}
