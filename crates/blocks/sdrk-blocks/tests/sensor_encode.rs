//! `SensorEncodeBlock` (M46 extension): the block equals the proto path (JSON, encoder, packets)
//! on whole buffers and over 20 chunk plans (seed 774), writes a reading whole or not at all,
//! counts what it cannot send, honours `repeats`, round-trips every device through
//! `SensorDecodeBlock`, and has the exact arena size.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::large_stack_arrays,
    clippy::items_after_statements
)]

mod common;

use sdrk_blocks::{
    Block, Error, Packet, ParamValue, SENSOR_ENCODE_PACKETS, SensorDecodeBlock, SensorDecodeConfig,
    SensorEncodeBlock, SensorEncodeConfig,
};
use sdrk_proto::PacketFlags;
use sdrk_proto::pulses::{self, CONTINUED, MAX_PAIRS, Pair};
use sdrk_proto::sensors::{Reading, TrainSink, encoder_by_model};
use sdrk_testkit::VecStorage;

/// The seed of the chunk plans (the block and the kind of M46's encoder extension).
const SEED_CHUNKS: u64 = 774;

/// One reading per model of the registry, as the JSON text of `sensor_decode`.
const READINGS: [&str; 7] = [
    r#"{"model":"wh2","id":93,"temperature_C":-3.5,"humidity":71}"#,
    r#"{"model":"nexus","id":92,"channel":2,"battery_ok":true,"temperature_C":-3.5,"humidity":71}"#,
    r#"{"model":"oregon_thn132n","id":123,"channel":1,"battery_ok":true,"temperature_C":21.3}"#,
    r#"{"model":"oregon_thgr122nx","id":45,"channel":2,"battery_ok":false,"temperature_C":-7.4,"humidity":55}"#,
    r#"{"model":"wh1080","id":123,"battery_ok":true,"temperature_C":18.5,"humidity":40,"wind_avg_km_h":12.2,"wind_max_km_h":24.5,"wind_dir_deg":90.0,"rain_mm":15.0}"#,
    r#"{"model":"lacrosse_tx29","id":45,"battery_ok":true,"temperature_C":12.3,"humidity":48}"#,
    r#"{"model":"Bresser-3CH","id":66,"channel":1,"battery_ok":true,"temperature_C":22.56,"humidity":66}"#,
];

fn block(cfg: SensorEncodeConfig) -> SensorEncodeBlock<VecStorage> {
    SensorEncodeBlock::new(&cfg, &mut VecStorage).unwrap()
}

fn text_packet(index: u64, text: &str) -> Packet {
    Packet::new(index, PacketFlags::NONE, text.as_bytes()).unwrap()
}

/// The packets of the proto path: JSON, the encoder of the model, the packets of the train.
fn proto_packets(index: u64, text: &str, repeats: u32) -> Vec<Packet> {
    let r = Reading::from_json(text).unwrap();
    let enc = encoder_by_model(r.model).unwrap();
    let mut storage = vec![Pair::default(); sdrk_proto::sensors::MAX_TRAIN_PAIRS];
    let mut sink = TrainSink::new(&mut storage);
    let repeats = if repeats == 0 {
        enc.default_repeats()
    } else {
        repeats
    };
    enc.encode(&r, repeats, &mut sink).unwrap();
    pulses::packets(index, PacketFlags::NONE, &enc.header(), sink.pairs())
        .map(|p| p.unwrap())
        .collect()
}

fn inputs() -> Vec<Packet> {
    READINGS
        .iter()
        .enumerate()
        .map(|(i, t)| text_packet(1000 * (i as u64 + 1), t))
        .collect()
}

fn wanted() -> Vec<Packet> {
    READINGS
        .iter()
        .enumerate()
        .flat_map(|(i, t)| proto_packets(1000 * (i as u64 + 1), t, 0))
        .collect()
}

/// Runs `input` through `b` with chunks of `sizes` packets and room for `room` packets.
fn run(
    b: &mut SensorEncodeBlock<VecStorage>,
    input: &[Packet],
    sizes: &mut dyn Iterator<Item = usize>,
    room: usize,
) -> Vec<Packet> {
    let mut out = Vec::new();
    let mut at = 0;
    let mut stalls = 0;
    while at < input.len() {
        let n = sizes.next().unwrap().min(input.len() - at);
        let mut buf = vec![Packet::default(); room];
        let p = b.process(&input[at..at + n], &mut buf).unwrap();
        out.extend_from_slice(&buf[..p.produced]);
        at += p.consumed;
        stalls = if p.consumed == 0 { stalls + 1 } else { 0 };
        assert!(stalls < 8, "no progress");
    }
    out
}

#[test]
fn the_block_equals_the_proto_path_on_whole_buffers_and_over_20_plans() {
    let want = wanted();
    assert!(want.len() > 60, "{}", want.len());
    let input = inputs();
    let mut b = block(SensorEncodeConfig::default());
    let got = run(&mut b, &input, &mut std::iter::repeat(usize::MAX), 4096);
    assert!(got == want, "whole buffer");
    assert_eq!(b.encoded(), 7);
    for (k, plan) in common::twenty_plans(SEED_CHUNKS).into_iter().enumerate() {
        // Room for one train at a time up to a few trains.
        let room = SENSOR_ENCODE_PACKETS + 5 * (k % 4);
        let mut b = block(SensorEncodeConfig::default());
        let got = run(&mut b, &input, &mut plan.sizes(), room);
        assert!(got == want, "plan {k}");
    }
}

#[test]
fn a_reading_is_written_whole_or_not_at_all() {
    let text = READINGS[6]; // bresser_3ch: 22 frames, about 37 packets
    let need = proto_packets(5, text, 0).len();
    assert!(need > 30 && need <= SENSOR_ENCODE_PACKETS, "{need}");
    let mut b = block(SensorEncodeConfig::default());
    let mut small = vec![Packet::default(); need - 1];
    let p = b.process(&[text_packet(5, text)], &mut small).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    assert_eq!(b.encoded(), 0);
    let mut exact = vec![Packet::default(); need];
    let p = b.process(&[text_packet(5, text)], &mut exact).unwrap();
    assert_eq!((p.consumed, p.produced), (1, need));
    assert_eq!(b.encoded(), 1);
    // The packets carry the index, all but the last the continued flag.
    assert!(exact.iter().all(|p| p.index == 5));
    assert!(
        exact[..need - 1]
            .iter()
            .all(|p| p.flags.contains(CONTINUED))
    );
    assert!(!exact[need - 1].flags.contains(CONTINUED));
    // A rejected reading in front of a full output is consumed (it needs no room).
    let mut none: [Packet; 0] = [];
    let p = b.process(&[text_packet(6, "not json")], &mut none).unwrap();
    assert_eq!((p.consumed, p.produced), (1, 0));
}

#[test]
fn what_cannot_be_sent_is_counted_and_consumed() {
    let mut b = block(SensorEncodeConfig::default());
    let bad_utf8 = Packet::new(0, PacketFlags::NONE, &[0xFF, 0xFE, b'{']).unwrap();
    let input = [
        text_packet(0, "not json"),
        bad_utf8,
        text_packet(1, r#"{"model":"nobody","id":1}"#),
        text_packet(
            2,
            r#"{"model":"wh2","id":1,"temperature_C":99.0,"humidity":10}"#,
        ),
        text_packet(
            3,
            r#"{"model":"wh2","id":1,"temperature_C":1.0,"humidity":10,"rain_mm":1}"#,
        ),
        text_packet(4, r#"{"model":"wh2","id":1,"humidity":10}"#),
        text_packet(5, READINGS[0]),
    ];
    let mut out = vec![Packet::default(); 64];
    let p = b.process(&input, &mut out).unwrap();
    assert_eq!(p.consumed, 7);
    assert!(p.produced > 0);
    assert_eq!(b.encoded(), 1);
    assert_eq!(b.invalid(), 2);
    assert_eq!(b.unknown_model(), 1);
    assert_eq!(b.rejected(), 3);
    assert_eq!(b.too_long(), 0);
    let names: Vec<_> = (0..)
        .map_while(|i| b.reading(i))
        .map(|(n, v)| (n, v as u64))
        .collect();
    assert_eq!(
        names,
        [
            ("encoded", 1),
            ("invalid", 2),
            ("unknown_model", 1),
            ("rejected", 3),
            ("too_long", 0)
        ]
    );
    // Reset clears the counters.
    b.reset();
    assert_eq!((b.encoded(), b.invalid(), b.rejected()), (0, 0, 0));
}

#[test]
fn repeats_set_the_copies_and_too_many_are_too_long() {
    let nexus = READINGS[1];
    // The device default is 4 frames of 37 pairs; 1 and 3 are asked for.
    for (repeats, pairs) in [(0u32, 4 * 37usize), (1, 37), (3, 3 * 37)] {
        let mut b = block(SensorEncodeConfig { repeats });
        let mut out = vec![Packet::default(); 64];
        let p = b.process(&[text_packet(0, nexus)], &mut out).unwrap();
        assert_eq!(p.produced, pairs.div_ceil(MAX_PAIRS), "repeats {repeats}");
        let last = pulses::decode(out[p.produced - 1].data()).unwrap();
        let rest = pairs - (p.produced - 1) * MAX_PAIRS;
        assert_eq!(last.len(), rest, "repeats {repeats}");
    }
    // Bresser: 27 frames fit 1200 pairs, 28 do not.
    let bresser = READINGS[6];
    for (repeats, ok) in [(27u32, true), (28, false)] {
        let mut b = block(SensorEncodeConfig { repeats });
        let mut out = vec![Packet::default(); 64];
        let p = b.process(&[text_packet(0, bresser)], &mut out).unwrap();
        assert_eq!(p.consumed, 1);
        assert_eq!(p.produced > 0, ok, "repeats {repeats}");
        assert_eq!(b.too_long(), u64::from(!ok), "repeats {repeats}");
    }
}

#[test]
fn the_trains_decode_to_the_readings() {
    // Every device through the encoder block and the decoder block.
    for text in READINGS {
        let input = text_packet(10, text);
        let mut enc = block(SensorEncodeConfig::default());
        let mut trains = vec![Packet::default(); 64];
        let p = enc.process(&[input], &mut trains).unwrap();
        assert_eq!(p.consumed, 1);
        let mut dec =
            SensorDecodeBlock::new(&SensorDecodeConfig::default(), &mut VecStorage).unwrap();
        let mut out = vec![Packet::default(); 8];
        let q = dec.process(&trains[..p.produced], &mut out).unwrap();
        assert_eq!((q.consumed, q.produced), (p.produced, 1), "{text}");
        assert_eq!(out[0].index, 10);
        let got = std::str::from_utf8(out[0].data()).unwrap();
        let want = Reading::from_json(text).unwrap();
        let back = Reading::from_json(got).unwrap();
        assert_eq!(back.model, want.model, "{text}");
        assert_eq!(back.id, want.id, "{text}");
        assert_eq!(back.channel, want.channel, "{text}");
        assert_eq!(back.battery_ok, want.battery_ok, "{text}");
        assert_eq!(back.humidity, want.humidity, "{text}");
        assert_eq!(back.temperature_c, want.temperature_c, "{text}");
        assert_eq!(back.rain_mm.is_some(), want.rain_mm.is_some(), "{text}");
        assert_eq!(back.wind_dir_deg, want.wind_dir_deg, "{text}");
    }
}

#[test]
fn the_arena_size_is_exact_and_the_struct_is_small() {
    let cfg = SensorEncodeConfig::default();
    let bytes =
        SensorEncodeBlock::<sdrk_blocks::ArenaStorage<'static>>::storage_bytes(&cfg).unwrap();
    assert_eq!(bytes, 1200 * 8);
    common::check_exact_storage(bytes, |arena| {
        SensorEncodeBlock::new(&cfg, arena).map(|_| ())
    });
    for size in [
        std::mem::size_of::<SensorEncodeBlock<VecStorage>>(),
        std::mem::size_of::<SensorEncodeBlock<sdrk_blocks::ArenaStorage<'static>>>(),
    ] {
        assert!(size <= 512, "{size}");
    }
    fn send_sync<T: Send + Sync>() {}
    send_sync::<SensorEncodeBlock<sdrk_blocks::ArenaStorage<'static>>>();
    send_sync::<SensorEncodeBlock<VecStorage>>();
}

#[test]
fn the_parameter_is_checked_and_read_only_at_run_time() {
    let mut cfg = SensorEncodeConfig::default();
    cfg.set("repeats", ParamValue::I64(22)).unwrap();
    assert_eq!(cfg.get("repeats").unwrap(), ParamValue::I64(22));
    assert!(matches!(
        cfg.set("repeats", ParamValue::I64(65)),
        Err(Error::OutOfRange {
            name: "repeats",
            ..
        })
    ));
    assert!(matches!(
        cfg.set("repeats", ParamValue::Bool(true)),
        Err(Error::WrongParamType)
    ));
    assert!(matches!(
        cfg.set("nope", ParamValue::I64(1)),
        Err(Error::UnknownParameter)
    ));
    let mut b = block(cfg);
    assert!(matches!(
        b.set_param("repeats", ParamValue::I64(3)),
        Err(Error::ReadOnlyParameter)
    ));
    assert_eq!(b.param("repeats").unwrap(), ParamValue::I64(22));
    let bad = SensorEncodeConfig { repeats: 65 };
    assert!(SensorEncodeBlock::<VecStorage>::new(&bad, &mut VecStorage).is_err());
}
