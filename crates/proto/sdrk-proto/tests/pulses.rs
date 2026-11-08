//! `sdrk_proto::pulses` (M45.1 spec test 2): the round trip of the payload, the version and
//! length errors, long trains over several packets and the JSON form.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use sdrk_device::{ErrorKind, SdrkError};
use sdrk_proto::pulses::{
    self, Assembler, CONTINUED, HEADER_BYTES, Header, MAX_PAIRS, Modulation, PAIR_BYTES, Pair,
    VERSION,
};
use sdrk_proto::{Error, MAX_PACKET_BYTES, Packet, PacketFlags};
use sdrk_testkit::rng::Pcg32;

fn header(rng: &mut Pcg32) -> Header {
    Header {
        modulation: if rng.below(2) == 0 {
            Modulation::Ook
        } else {
            Modulation::Fsk
        },
        channel: rng.next_u32() as i16,
        freq_offset_hz: rng.next_u32() as i32,
        level_db10: rng.next_u32() as i16,
        noise_db10: rng.next_u32() as i16,
        rate_hz: rng.next_u32(),
    }
}

fn pairs(rng: &mut Pcg32, n: usize) -> Vec<Pair> {
    (0..n)
        .map(|_| Pair {
            pulse: rng.next_u32(),
            gap: rng.next_u32(),
        })
        .collect()
}

#[test]
fn the_payload_round_trips_for_every_length_and_extreme_values() {
    let mut rng = Pcg32::new(2, 45);
    for n in 0..=MAX_PAIRS {
        for _ in 0..8 {
            let h = header(&mut rng);
            let p = pairs(&mut rng, n);
            let packet = pulses::encode_packet(rng.next_u64(), PacketFlags::NONE, &h, &p).unwrap();
            assert_eq!(packet.len(), pulses::payload_len(n));
            let t = pulses::decode(packet.data()).unwrap();
            assert_eq!(t.header(), h);
            assert_eq!(t.len(), n);
            assert_eq!(t.is_empty(), n == 0);
            assert_eq!(t.pairs().len(), n);
            assert_eq!(t.pairs().collect::<Vec<_>>(), p);
        }
    }
    let extreme = Header {
        modulation: Modulation::Fsk,
        channel: i16::MIN,
        freq_offset_hz: i32::MIN,
        level_db10: i16::MAX,
        noise_db10: i16::MIN,
        rate_hz: u32::MAX,
    };
    let p = [Pair {
        pulse: u32::MAX,
        gap: 0,
    }];
    let packet = pulses::encode_packet(u64::MAX, PacketFlags::NONE, &extreme, &p).unwrap();
    let t = pulses::decode(packet.data()).unwrap();
    assert_eq!(t.header(), extreme);
    assert_eq!(t.pairs().next(), Some(p[0]));
}

#[test]
fn the_byte_layout_is_little_endian() {
    let h = Header {
        modulation: Modulation::Fsk,
        channel: -2,
        freq_offset_hz: -250_000,
        level_db10: -153,
        noise_db10: -421,
        rate_hz: 250_000,
    };
    let mut buf = [0u8; 64];
    let len = pulses::encode(
        &h,
        &[Pair {
            pulse: 0x0102_0304,
            gap: 7,
        }],
        &mut buf,
    )
    .unwrap();
    assert_eq!(len, HEADER_BYTES + PAIR_BYTES);
    let mut expected = vec![VERSION, 1];
    expected.extend_from_slice(&(-2i16).to_le_bytes());
    expected.extend_from_slice(&(-250_000i32).to_le_bytes());
    expected.extend_from_slice(&(-153i16).to_le_bytes());
    expected.extend_from_slice(&(-421i16).to_le_bytes());
    expected.extend_from_slice(&250_000u32.to_le_bytes());
    expected.extend_from_slice(&[4, 3, 2, 1, 7, 0, 0, 0]);
    assert_eq!(&buf[..len], expected.as_slice());
    assert_eq!(Modulation::Ook.code(), 0);
    assert_eq!(Modulation::from_code(1), Ok(Modulation::Fsk));
}

#[test]
fn version_and_length_errors() {
    let h = Header {
        modulation: Modulation::Ook,
        channel: 3,
        freq_offset_hz: 375_000,
        level_db10: -100,
        noise_db10: -400,
        rate_hz: 250_000,
    };
    let good =
        pulses::encode_packet(0, PacketFlags::NONE, &h, &[Pair { pulse: 1, gap: 2 }; 3]).unwrap();
    let data = good.data().to_vec();
    // Shorter than the header.
    for len in 0..HEADER_BYTES {
        let e = pulses::decode(&data[..len]).unwrap_err();
        assert_eq!(e, Error::Parse("pulse train shorter than its header"));
        assert_eq!(e.kind(), ErrorKind::Parse);
    }
    // Not a whole number of pairs.
    for cut in 1..PAIR_BYTES {
        assert_eq!(
            pulses::decode(&data[..data.len() - cut]),
            Err(Error::Parse("pulse train is not a header and whole pairs"))
        );
    }
    // Every other version.
    for v in (0..=255u8).filter(|&v| v != VERSION) {
        let mut d = data.clone();
        d[0] = v;
        assert_eq!(
            pulses::decode(&d),
            Err(Error::Parse("unknown pulse train version"))
        );
    }
    // Unknown modulation.
    let mut d = data.clone();
    d[1] = 2;
    assert_eq!(
        pulses::decode(&d),
        Err(Error::Parse("unknown pulse train modulation"))
    );
    // Encoding: more pairs than a packet holds, and a short buffer.
    let many = [Pair::default(); MAX_PAIRS + 1];
    let e = pulses::encode_packet(0, PacketFlags::NONE, &h, &many).unwrap_err();
    assert_eq!(
        e,
        Error::TooLong {
            len: HEADER_BYTES + (MAX_PAIRS + 1) * PAIR_BYTES,
            max: MAX_PACKET_BYTES
        }
    );
    assert_eq!(e.kind(), ErrorKind::OutOfRange);
    let mut small = [0u8; HEADER_BYTES + PAIR_BYTES - 1];
    assert_eq!(
        pulses::encode(&h, &[Pair::default()], &mut small),
        Err(Error::BufferTooSmall {
            needed: HEADER_BYTES + PAIR_BYTES,
            got: HEADER_BYTES + PAIR_BYTES - 1
        })
    );
    assert_eq!(MAX_PAIRS, 30);
    assert_eq!(pulses::payload_len(MAX_PAIRS), 256);
}

#[test]
fn long_trains_are_split_and_joined_again() {
    let mut rng = Pcg32::new(2, 46);
    let mut asm = Assembler::new(vec![Pair::default(); 1200]);
    for n in [0, 1, 29, 30, 31, 60, 61, 1200] {
        let h = header(&mut rng);
        let p = pairs(&mut rng, n);
        let parts: Vec<Packet> = pulses::packets(77, PacketFlags::TRUNCATED, &h, &p)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(parts.len(), n.div_ceil(MAX_PAIRS).max(1), "{n} pairs");
        for (k, part) in parts.iter().enumerate() {
            assert_eq!(part.index, 77);
            assert!(part.flags.contains(PacketFlags::TRUNCATED));
            assert_eq!(part.flags.contains(CONTINUED), k + 1 < parts.len());
            assert_eq!(pulses::decode(part.data()).unwrap().header(), h);
        }
        let mut out = None;
        for (k, part) in parts.iter().enumerate() {
            let r = asm.push(part).unwrap();
            assert_eq!(r.is_some(), k + 1 == parts.len());
            if let Some(t) = r {
                out = Some((t.index, t.flags, t.header, t.pairs.to_vec()));
            }
        }
        assert_eq!(out, Some((77, PacketFlags::TRUNCATED, h, p)));
    }
}

#[test]
fn the_assembler_truncates_restarts_and_drops_bad_packets() {
    let h = Header {
        modulation: Modulation::Ook,
        channel: 0,
        freq_offset_hz: 0,
        level_db10: 0,
        noise_db10: 0,
        rate_hz: 250_000,
    };
    let p: Vec<Pair> = (0..50).map(|k| Pair { pulse: k, gap: k }).collect();
    let parts: Vec<Packet> = pulses::packets(5, PacketFlags::NONE, &h, &p)
        .collect::<Result<_, _>>()
        .unwrap();
    // Storage for 40 pairs: the train is cut and marked.
    let mut asm = Assembler::new([Pair::default(); 40]);
    assert!(asm.push(&parts[0]).unwrap().is_none());
    let t = asm.push(&parts[1]).unwrap().unwrap();
    assert_eq!(t.pairs, &p[..40]);
    assert_eq!(t.flags, PacketFlags::TRUNCATED);
    // A lost last part: the next train (another index) starts afresh.
    assert!(asm.push(&parts[0]).unwrap().is_none());
    let single = pulses::encode_packet(6, PacketFlags::CRC_OK, &h, &p[..3]).unwrap();
    let t = asm.push(&single).unwrap().unwrap();
    assert_eq!(
        (t.index, t.flags, t.pairs),
        (6, PacketFlags::CRC_OK, &p[..3])
    );
    // A malformed packet is an error and drops the train in progress.
    assert!(asm.push(&parts[0]).unwrap().is_none());
    let bad = Packet::new(5, CONTINUED, &[0u8; 3]).unwrap();
    assert!(asm.push(&bad).is_err());
    let t = asm.push(&parts[1]).unwrap().unwrap();
    assert_eq!(t.pairs, &p[30..]);
    // `reset` drops a partial train too.
    assert!(asm.push(&parts[0]).unwrap().is_none());
    asm.reset();
    assert_eq!(asm.push(&parts[1]).unwrap().unwrap().pairs, &p[30..]);
}

#[test]
fn the_json_form() {
    let mut h = Header {
        modulation: Modulation::Fsk,
        channel: 2,
        freq_offset_hz: 312_345,
        level_db10: -5,
        noise_db10: 0,
        rate_hz: 500_000,
    };
    let mut s = String::new();
    pulses::write_json(
        &mut s,
        &h,
        [Pair { pulse: 1, gap: 2 }, Pair { pulse: 30, gap: 4 }],
        868_500_000,
    )
    .unwrap();
    assert_eq!(
        s,
        r#"{"mod":"fsk","freq_hz":868812345,"rssi_db":-0.5,"noise_db":0.0,"channel":2,"rate_hz":500000,"pulses":[[1,2],[30,4]]}"#
    );
    h.level_db10 = i16::MIN;
    h.noise_db10 = 123;
    h.modulation = Modulation::Ook;
    let mut s = String::new();
    pulses::write_json(&mut s, &h, [], 0).unwrap();
    assert_eq!(
        s,
        r#"{"mod":"ook","freq_hz":312345,"rssi_db":-3276.8,"noise_db":12.3,"channel":2,"rate_hz":500000,"pulses":[]}"#
    );
}

#[test]
fn types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Header>();
    assert_send_sync::<Pair>();
    assert_send_sync::<Modulation>();
    assert_send_sync::<pulses::PulseTrain<'static>>();
    assert_send_sync::<pulses::Pairs<'static>>();
    assert_send_sync::<pulses::Packets<'static>>();
    assert_send_sync::<Assembler<[Pair; 4]>>();
    assert_send_sync::<pulses::Assembled<'static>>();
}
