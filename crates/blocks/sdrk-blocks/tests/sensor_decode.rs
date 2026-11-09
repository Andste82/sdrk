//! `SensorDecodeBlock` (M46 §4): the block equals the decoders on whole buffers (the golden trains
//! of the Python encoders), is chunk invariant with 20 plans (test 6, seed 643), merges the copies
//! of a reading (same radio channel within `merge_s`, another channel within 0.5 s), writes the
//! unknown trains with `emit_unknown` (test 5, seed 642), joins trains of several packets, and
//! has the exact arena size.
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

use sdrk_blocks::{
    AlignedBytes, ArenaStorage, Block, Error, Packet, ParamValue, SENSOR_ALL_DECODERS,
    SENSOR_TRAIN_PAIRS, SensorDecodeBlock, SensorDecodeConfig,
};
use sdrk_proto::PacketFlags;
use sdrk_proto::pulses::{self, Header, Modulation, Pair};
use sdrk_proto::sensors::{self, DECODERS, ReadingSink};
use sdrk_testkit::golden::{self, Array};
use sdrk_testkit::rng::Pcg32;
use sdrk_testkit::{ChunkPlan, VecStorage};

const RATE: u32 = 250_000;
const SEED_UNKNOWN: u64 = 642;
const SEED_CHUNKS: u64 = 643;
const CASES: [&str; 6] = [
    "wh2",
    "nexus",
    "oregon_v21",
    "wh1080",
    "lacrosse_tx29",
    "bresser_3ch",
];

fn header(m: Modulation, channel: i16) -> Header {
    Header {
        modulation: m,
        channel,
        freq_offset_hz: 12_500,
        level_db10: -250,
        noise_db10: -600,
        rate_hz: RATE,
    }
}

fn block(cfg: &SensorDecodeConfig) -> SensorDecodeBlock<VecStorage> {
    SensorDecodeBlock::new(cfg, &mut VecStorage).unwrap()
}

/// The pairs of every train of a golden case.
fn golden_trains(name: &str) -> Vec<Vec<Pair>> {
    let c = golden::load("sensors", name).unwrap();
    let Array::I32(counts) = c.array("pair_counts").unwrap() else {
        panic!("pair_counts");
    };
    let Array::I32(flat) = c.array("pairs").unwrap() else {
        panic!("pairs");
    };
    let mut at = 0;
    let mut out = Vec::new();
    for &n in counts {
        let n = n as usize;
        out.push(
            flat[at..at + 2 * n]
                .chunks_exact(2)
                .map(|w| Pair {
                    pulse: w[0] as u32,
                    gap: w[1] as u32,
                })
                .collect(),
        );
        at += 2 * n;
    }
    out
}

fn modulation_of(name: &str) -> Modulation {
    sensors::by_name(name).unwrap().modulation()
}

/// The packets of a train (several when it has more than 30 pairs).
fn packets_of(index: u64, h: &Header, pairs: &[Pair]) -> Vec<Packet> {
    pulses::packets(index, PacketFlags::NONE, h, pairs)
        .map(|p| p.unwrap())
        .collect()
}

/// Runs `packets` through `block` with chunks and output room from the iterators.
fn run(
    block: &mut SensorDecodeBlock<VecStorage>,
    packets: &[Packet],
    chunks: &mut dyn Iterator<Item = usize>,
    room: &mut dyn Iterator<Item = usize>,
) -> Vec<Packet> {
    let mut out = Vec::new();
    let mut at = 0;
    let mut stalls = 0;
    while at < packets.len() {
        let n = chunks.next().unwrap().min(packets.len() - at);
        // The block needs room for 4 packets before it reads a train.
        let r = room.next().unwrap().max(4);
        let mut buf = vec![Packet::default(); r];
        let p = block.process(&packets[at..at + n], &mut buf).unwrap();
        assert!(p.consumed <= n && p.produced <= r);
        out.extend_from_slice(&buf[..p.produced]);
        at += p.consumed;
        stalls = if p.consumed == 0 { stalls + 1 } else { 0 };
        assert!(stalls < 8, "no progress");
    }
    out
}

fn texts(packets: &[Packet]) -> Vec<String> {
    packets
        .iter()
        .map(|p| String::from_utf8(p.data().to_vec()).unwrap())
        .collect()
}

/// What the decoders alone give for a train, in registry order, without readings that repeat an
/// earlier one of the same train, as the JSON the block writes.
fn expected_json(index: u64, h: &Header, pairs: &[Pair], center_hz: u32) -> Vec<String> {
    let mut buf = vec![0u8; 16 + 8 * pairs.len()];
    let n = pulses::encode_joined(h, pairs, &mut buf).unwrap();
    let train = pulses::decode(&buf[..n]).unwrap();
    let mut got: Vec<sensors::Reading> = Vec::new();
    for d in DECODERS {
        if d.modulation() != h.modulation {
            continue;
        }
        let mut sink = ReadingSink::new();
        d.decode(&train, &mut sink);
        for r in sink.iter() {
            if got.iter().any(|g| g.same_content(r)) {
                continue;
            }
            let mut r = *r;
            r.time_s = Some((index as f64 / f64::from(h.rate_hz) * 1000.0).round() / 1000.0);
            r.center_hz = center_hz;
            got.push(r);
        }
    }
    got.iter()
        .map(|r| {
            let mut s = String::new();
            r.to_json(&mut s).unwrap();
            s
        })
        .collect()
}

#[test]
fn the_block_equals_the_decoders_on_whole_buffers() {
    let cfg = SensorDecodeConfig {
        center_hz: 433_920_000,
        ..SensorDecodeConfig::default()
    };
    let mut b = block(&cfg);
    let mut want = Vec::new();
    let mut index = 0u64;
    let mut stream = Vec::new();
    for name in CASES {
        let n = golden_trains(name).len();
        assert_eq!(n, 20, "{name}");
        for pairs in golden_trains(name) {
            let h = header(modulation_of(name), 0);
            want.extend(expected_json(index, &h, &pairs, cfg.center_hz));
            stream.extend(packets_of(index, &h, &pairs));
            index += 4 * u64::from(RATE);
        }
    }
    assert!(want.len() >= 100, "{}", want.len());
    let got = run(
        &mut b,
        &stream,
        &mut std::iter::repeat(usize::MAX),
        &mut std::iter::repeat(1 << 16),
    );
    // The time is rounded to milliseconds by both sides; compare the texts.
    assert_eq!(texts(&got), want);
    for p in &got {
        assert!(p.data().len() <= 256);
    }
    assert_eq!(b.unknown(), 0);
    assert_eq!(b.invalid(), 0);
    assert_eq!(b.lost(), 0);
}

/// A stream with copies of readings: same channel at 0.1 s and 0.3 s, another channel at 0.2 s,
/// plus trains of several packets and trains that no decoder reads (to be written with
/// `emit_unknown`).
fn busy_stream() -> Vec<Packet> {
    let mut rng = Pcg32::new(SEED_CHUNKS, 9);
    let mut out = Vec::new();
    let mut index = 0u64;
    for name in CASES {
        let trains = golden_trains(name);
        for pairs in trains.iter().take(8) {
            let m = modulation_of(name);
            out.extend(packets_of(index, &header(m, 3), pairs));
            out.extend(packets_of(index + 25_000, &header(m, 3), pairs));
            out.extend(packets_of(index + 50_000, &header(m, -2), pairs));
            out.extend(packets_of(index + 75_000, &header(m, 3), pairs));
            // A train of random widths in between.
            let junk: Vec<Pair> = (0..=rng.below(90))
                .map(|_| Pair {
                    pulse: 5 + rng.below(400),
                    gap: 5 + rng.below(900),
                })
                .collect();
            out.extend(packets_of(index + 100_000, &header(m, 1), &junk));
            index += 3 * u64::from(RATE);
        }
    }
    out
}

#[test]
fn the_block_is_chunk_invariant_with_20_plans() {
    let stream = busy_stream();
    assert!(stream.iter().any(|p| p.flags.contains(pulses::CONTINUED)));
    let cfg = SensorDecodeConfig {
        emit_unknown: true,
        ..SensorDecodeConfig::default()
    };
    let mut b = block(&cfg);
    let want = run(
        &mut b,
        &stream,
        &mut std::iter::repeat(usize::MAX),
        &mut std::iter::repeat(1 << 16),
    );
    assert!(want.len() > 40, "{}", want.len());
    assert!(
        b.merged() > 0 && b.unknown() > 0,
        "{} {}",
        b.merged(),
        b.unknown()
    );
    let reference = (b.merged(), b.unknown(), b.invalid(), b.lost());
    for k in 0..20u64 {
        let chunks = ChunkPlan::new(SEED_CHUNKS, 1 + 3 * k as usize);
        let room = ChunkPlan::new(SEED_CHUNKS + 1, 4 + k as usize);
        b.reset();
        let got = run(&mut b, &stream, &mut chunks.sizes(), &mut room.sizes());
        assert!(got == want, "plan {k}");
        assert_eq!((b.merged(), b.unknown(), b.invalid(), b.lost()), reference);
    }
}

/// A Nexus train of a fixed reading (PPM, 12 equal frames of 36 bits with 4 ms sync gaps).
fn nexus_train() -> Vec<Pair> {
    let c = golden_trains("nexus");
    c[0].clone()
}

fn count_readings(cfg: &SensorDecodeConfig, trains: &[(u64, i16)]) -> Vec<String> {
    let pairs = nexus_train();
    let mut stream = Vec::new();
    for &(index, ch) in trains {
        stream.extend(packets_of(index, &header(Modulation::Ook, ch), &pairs));
    }
    let mut b = block(cfg);
    texts(&run(
        &mut b,
        &stream,
        &mut std::iter::repeat(usize::MAX),
        &mut std::iter::repeat(64),
    ))
}

#[test]
fn copies_are_merged_by_channel_and_time() {
    let cfg = SensorDecodeConfig::default();
    let s = u64::from(RATE);
    // Same channel: the copies within 2 s of the first are dropped, the one at 2.5 s is new.
    let got = count_readings(
        &cfg,
        &[(0, 1), (s / 10, 1), (s * 19 / 10, 1), (s * 5 / 2, 1)],
    );
    assert_eq!(got.len(), 2, "{got:?}");
    assert!(got[0].contains("\"time\":0.000") && got[1].contains("\"time\":2.500"));
    // Another channel: within 0.5 s it is the same transmission, later it is not.
    let got = count_readings(&cfg, &[(0, 1), (s * 4 / 10, 2), (s * 6 / 10, 3)]);
    assert_eq!(got.len(), 2, "{got:?}");
    assert!(got[1].contains("\"time\":0.600"));
    // merge_s 0 keeps every copy on the same channel.
    let none = SensorDecodeConfig {
        merge_s: 0.0,
        ..cfg
    };
    let got = count_readings(&none, &[(0, 1), (s / 10, 1), (s / 5, 1)]);
    assert_eq!(got.len(), 3, "{got:?}");
    // The merged copies are counted.
    let mut b = block(&cfg);
    let pairs = nexus_train();
    let mut stream = packets_of(0, &header(Modulation::Ook, 1), &pairs);
    stream.extend(packets_of(100, &header(Modulation::Ook, 1), &pairs));
    stream.extend(packets_of(200, &header(Modulation::Ook, 4), &pairs));
    let out = run(
        &mut b,
        &stream,
        &mut std::iter::repeat(usize::MAX),
        &mut std::iter::repeat(8),
    );
    assert_eq!(out.len(), 1);
    assert_eq!(b.merged(), 2);
    assert_eq!(b.reading(0), Some(("merged", 2.0)));
    assert_eq!(b.reading(4), None);
}

#[test]
fn a_reading_carries_the_radio_fields_and_the_index() {
    let pairs = nexus_train();
    let h = Header {
        modulation: Modulation::Ook,
        channel: 5,
        freq_offset_hz: -20_000,
        level_db10: -125,
        noise_db10: -600,
        rate_hz: RATE,
    };
    let cfg = SensorDecodeConfig {
        center_hz: 433_920_000,
        ..SensorDecodeConfig::default()
    };
    let mut b = block(&cfg);
    let mut out = [Packet::default(); 4];
    let train = packets_of(123_456, &h, &pairs);
    let train_len = train.len();
    let p = b.process(&train, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (train_len, 1));
    assert_eq!(out[0].index, 123_456);
    let s = String::from_utf8(out[0].data().to_vec()).unwrap();
    assert!(s.starts_with("{\"time\":0.494,\"model\":\"nexus\""), "{s}");
    assert!(s.contains("\"mic\":\"repeat\""), "{s}");
    assert!(
        s.contains("\"freq_MHz\":433.9000,\"rssi_dB\":-12.5,"),
        "{s}"
    );
    assert!(s.ends_with('}'));
    assert!(!out[0].flags.contains(PacketFlags::CRC_OK));
    // A CRC device sets the flag.
    let wh2 = golden_trains("wh2")[0].clone();
    let p = b
        .process(&packets_of(0, &header(Modulation::Ook, 0), &wh2), &mut out)
        .unwrap();
    assert_eq!(p.produced, 1);
    assert!(out[0].flags.contains(PacketFlags::CRC_OK));
}

#[test]
fn the_decoder_mask_selects_decoders() {
    let pairs = nexus_train();
    let stream = packets_of(0, &header(Modulation::Ook, 0), &pairs);
    let only = |names: &[&str]| SensorDecodeConfig {
        decoders: SensorDecodeConfig::mask_of(names).unwrap() as u32,
        ..SensorDecodeConfig::default()
    };
    for (names, n) in [
        (&["nexus"][..], 1),
        (&["wh2", "wh1080"][..], 0),
        (&["lacrosse_tx29"][..], 0),
    ] {
        let mut b = block(&only(names));
        let mut out = [Packet::default(); 4];
        assert_eq!(
            b.process(&stream, &mut out).unwrap().produced,
            n,
            "{names:?}"
        );
    }
    // An FSK train is not offered to an OOK decoder.
    let fsk = golden_trains("lacrosse_tx29")[0].clone();
    let mut b = block(&only(&["wh2", "nexus", "oregon_v21", "wh1080"]));
    let mut out = [Packet::default(); 4];
    let train = packets_of(0, &header(Modulation::Fsk, 0), &fsk);
    assert_eq!(b.process(&train, &mut out).unwrap().produced, 0);
    let mut b = block(&only(&["lacrosse_tx29"]));
    assert_eq!(b.process(&train, &mut out).unwrap().produced, 1);
}

fn is_balanced_json(s: &str) -> bool {
    let (mut depth, mut quote) = (0i32, false);
    for c in s.chars() {
        match c {
            '"' => quote = !quote,
            '{' | '[' if !quote => depth += 1,
            '}' | ']' if !quote => depth -= 1,
            _ => {}
        }
        if depth < 0 {
            return false;
        }
    }
    depth == 0 && !quote
}

/// Test 5 (seed 642): trains that no decoder accepts are written only with `emit_unknown`, with
/// their widths, cut to one packet.
#[test]
fn emit_unknown_writes_the_trains_no_decoder_accepts() {
    let mut rng = Pcg32::new(SEED_UNKNOWN, 1);
    let mut trains = Vec::new();
    for _ in 0..40 {
        let n = 1 + rng.below(120) as usize;
        let pairs: Vec<Pair> = (0..n)
            .map(|_| Pair {
                pulse: 1 + rng.below(2000),
                gap: 1 + rng.below(4000),
            })
            .collect();
        let m = if rng.below(2) == 0 {
            Modulation::Ook
        } else {
            Modulation::Fsk
        };
        trains.push((m, pairs));
    }
    let mut stream = Vec::new();
    for (i, (m, pairs)) in trains.iter().enumerate() {
        stream.extend(packets_of(i as u64 * 1000, &header(*m, 0), pairs));
    }
    let run_with = |emit: bool| {
        let cfg = SensorDecodeConfig {
            emit_unknown: emit,
            ..SensorDecodeConfig::default()
        };
        let mut b = block(&cfg);
        let out = run(
            &mut b,
            &stream,
            &mut std::iter::repeat(usize::MAX),
            &mut std::iter::repeat(64),
        );
        (out, b.unknown())
    };
    let (quiet, n) = run_with(false);
    assert!(quiet.is_empty() && n == 0);
    let (out, n) = run_with(true);
    assert_eq!(out.len(), trains.len(), "random trains are not readings");
    assert_eq!(n, trains.len() as u64);
    for ((m, pairs), p) in trains.iter().zip(&out) {
        let s = std::str::from_utf8(p.data()).unwrap();
        assert!(s.contains("\"model\":\"unknown\""), "{s}");
        assert!(s.contains(&format!("\"mod\":\"{}\"", m.as_str())), "{s}");
        assert!(s.contains(&format!("\"pairs\":{}", pairs.len())), "{s}");
        assert!(is_balanced_json(s) && s.ends_with("]}"), "{s}");
        assert!(s.contains("\"rssi_dB\":-25.0"), "{s}");
        // The widths come first and exactly.
        let first = format!("\"pulses\":[[{},{}]", pairs[0].pulse, pairs[0].gap);
        assert!(s.contains(&first), "{s}");
        let cut = p.flags.contains(PacketFlags::TRUNCATED);
        let listed = s.matches('[').count() - 1;
        assert_eq!(cut, listed < pairs.len(), "{s}");
        assert!(listed <= pairs.len());
    }
    // A decoded train is not unknown.
    let mut b = block(&SensorDecodeConfig {
        emit_unknown: true,
        ..SensorDecodeConfig::default()
    });
    let mut out = [Packet::default(); 4];
    let wh2 = packets_of(0, &header(Modulation::Ook, 0), &golden_trains("wh2")[0]);
    b.process(&wh2, &mut out).unwrap();
    assert!(!String::from_utf8_lossy(out[0].data()).contains("unknown"));
    assert_eq!(b.unknown(), 0);
}

#[test]
fn a_train_of_several_packets_is_joined_and_a_lost_part_drops_it() {
    let cfg = SensorDecodeConfig {
        emit_unknown: true,
        ..SensorDecodeConfig::default()
    };
    let pairs: Vec<Pair> = (0..100)
        .map(|k| Pair {
            pulse: 10 + k,
            gap: 30,
        })
        .collect();
    let h = header(Modulation::Ook, 0);
    let all = packets_of(7, &h, &pairs);
    assert_eq!(all.len(), 4);
    let mut b = block(&cfg);
    let mut out = [Packet::default(); 8];
    // One packet at a time: nothing until the last.
    for (k, p) in all.iter().enumerate() {
        let r = b.process(std::slice::from_ref(p), &mut out).unwrap();
        assert_eq!((r.consumed, r.produced), (1, usize::from(k == 3)), "{k}");
    }
    let s = std::str::from_utf8(out[0].data()).unwrap();
    assert!(s.contains("\"pairs\":100"), "{s}");
    // A train that lost its middle: the part with another index starts a new train.
    let mut b = block(&cfg);
    let mut got = 0;
    for p in [&all[0], &all[3]] {
        got += b
            .process(std::slice::from_ref(p), &mut out)
            .unwrap()
            .produced;
    }
    assert_eq!(got, 1, "the last part alone still gives a (short) train");
    let s = std::str::from_utf8(out[0].data()).unwrap();
    assert!(s.contains("\"pairs\":"), "{s}");
}

#[test]
fn a_train_beyond_the_storage_is_cut_and_decoded_as_far_as_it_goes() {
    let cfg = SensorDecodeConfig {
        emit_unknown: true,
        ..SensorDecodeConfig::default()
    };
    let pairs: Vec<Pair> = (0..SENSOR_TRAIN_PAIRS + 50)
        .map(|k| Pair {
            pulse: 10 + (k as u32 % 7),
            gap: 30,
        })
        .collect();
    let mut b = block(&cfg);
    let mut out = [Packet::default(); 8];
    let stream = packets_of(0, &header(Modulation::Ook, 0), &pairs);
    let p = b.process(&stream, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (stream.len(), 1));
    assert!(out[0].flags.contains(PacketFlags::TRUNCATED));
    let s = std::str::from_utf8(out[0].data()).unwrap();
    assert!(
        s.contains(&format!("\"pairs\":{SENSOR_TRAIN_PAIRS}")),
        "{s}"
    );
}

#[test]
fn garbage_packets_are_counted_and_dropped() {
    let mut b = block(&SensorDecodeConfig::default());
    let junk = Packet::new(0, PacketFlags::NONE, &[1, 2, 3]).unwrap();
    let mut out = [Packet::default(); 4];
    let p = b.process(&[junk, junk], &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (2, 0));
    assert_eq!(b.invalid(), 2);
    assert_eq!(b.reading(2), Some(("invalid", 2.0)));
}

#[test]
fn the_block_waits_for_room_for_a_whole_train() {
    let mut b = block(&SensorDecodeConfig::default());
    let train = packets_of(0, &header(Modulation::Ook, 0), &nexus_train());
    let mut small = [Packet::default(); 3];
    // The first parts of a train give nothing, the last one waits for room.
    let p = b.process(&train, &mut small).unwrap();
    assert_eq!((p.consumed, p.produced), (train.len() - 1, 0));
    let rest = &train[train.len() - 1..];
    let p = b.process(rest, &mut small).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    let mut out = [Packet::default(); 4];
    let p = b.process(rest, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (1, 1));
}

#[test]
fn parameters_are_validated_and_read_only() {
    let mut cfg = SensorDecodeConfig::default();
    assert_eq!(SENSOR_ALL_DECODERS, (1 << DECODERS.len()) - 1);
    assert_eq!(
        SensorDecodeConfig::mask_of(&[
            "wh2",
            "nexus",
            "oregon_v21",
            "wh1080",
            "lacrosse_tx29",
            "bresser_3ch",
        ])
        .unwrap(),
        SENSOR_ALL_DECODERS
    );
    assert!(matches!(
        cfg.set("decoders", ParamValue::I64(0)),
        Err(Error::OutOfRange {
            name: "decoders",
            ..
        })
    ));
    assert!(matches!(
        cfg.set("decoders", ParamValue::I64(1 << 6)),
        Err(Error::OutOfRange {
            name: "decoders",
            ..
        })
    ));
    assert!(matches!(
        cfg.set("merge_s", ParamValue::F64(-1.0)),
        Err(Error::OutOfRange {
            name: "merge_s",
            ..
        })
    ));
    assert!(matches!(
        cfg.set("merge_s", ParamValue::Bool(true)),
        Err(Error::WrongParamType)
    ));
    assert!(matches!(
        cfg.set("nope", ParamValue::Bool(true)),
        Err(Error::UnknownParameter)
    ));
    assert!(matches!(
        SensorDecodeConfig::mask_of(&["rtl"]),
        Err(Error::InvalidParameter {
            name: "decoders",
            ..
        })
    ));
    cfg.set("merge_s", ParamValue::F64(5.0)).unwrap();
    cfg.set("emit_unknown", ParamValue::Bool(true)).unwrap();
    cfg.set("center_hz", ParamValue::I64(868_300_000)).unwrap();
    let mut b = block(&cfg);
    assert_eq!(b.param("merge_s").unwrap(), ParamValue::F64(5.0));
    assert_eq!(b.param("center_hz").unwrap(), ParamValue::I64(868_300_000));
    assert!(matches!(
        b.set_param("merge_s", ParamValue::F64(1.0)),
        Err(Error::ReadOnlyParameter)
    ));
    assert!(matches!(
        b.set_param("nope", ParamValue::F64(1.0)),
        Err(Error::UnknownParameter)
    ));
    let bad = SensorDecodeConfig {
        decoders: 0,
        ..SensorDecodeConfig::default()
    };
    assert!(SensorDecodeBlock::<VecStorage>::new(&bad, &mut VecStorage).is_err());
}

#[test]
fn the_arena_size_is_exact_and_the_struct_is_small() {
    let cfg = SensorDecodeConfig::default();
    let bytes = SensorDecodeBlock::<ArenaStorage<'_>>::storage_bytes(&cfg).unwrap();
    let mut mem = AlignedBytes([0u8; 12288]);
    assert!(bytes <= mem.0.len());
    let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
    let b = SensorDecodeBlock::new(&cfg, &mut arena).unwrap();
    assert_eq!(arena.used(), bytes);
    assert_eq!(arena.remaining(), 0);
    assert_eq!(b.merged(), 0);
    let mut small = AlignedBytes([0u8; 12288]);
    let mut arena = ArenaStorage::new(&mut small.0[..bytes - 1]);
    assert!(matches!(
        SensorDecodeBlock::new(&cfg, &mut arena),
        Err(Error::Storage { .. })
    ));
    for size in [
        std::mem::size_of::<SensorDecodeBlock<VecStorage>>(),
        std::mem::size_of::<SensorDecodeBlock<ArenaStorage<'static>>>(),
    ] {
        assert!(size <= 512, "{size}");
    }
    fn send_sync<T: Send + Sync>() {}
    send_sync::<SensorDecodeBlock<ArenaStorage<'static>>>();
    send_sync::<SensorDecodeBlock<VecStorage>>();
    send_sync::<SensorDecodeConfig>();
}

#[test]
fn the_block_also_runs_from_an_arena_and_reset_forgets_the_copies() {
    let cfg = SensorDecodeConfig::default();
    let bytes = SensorDecodeBlock::<ArenaStorage<'_>>::storage_bytes(&cfg).unwrap();
    let mut mem = AlignedBytes([0u8; 12288]);
    let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
    let mut b = SensorDecodeBlock::new(&cfg, &mut arena).unwrap();
    let train = packets_of(0, &header(Modulation::Ook, 0), &nexus_train());
    let mut out = [Packet::default(); 4];
    assert_eq!(b.process(&train, &mut out).unwrap().produced, 1);
    assert_eq!(b.process(&train, &mut out).unwrap().produced, 0);
    b.reset();
    assert_eq!(b.process(&train, &mut out).unwrap().produced, 1);
    assert_eq!(b.merged(), 0);
    let t = b.timing();
    assert_eq!(
        t.kind,
        sdrk::RatioKind::Packets {
            max_per_input: t.ratio
        }
    );
}
