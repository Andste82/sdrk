//! NRZI and HDLC of `sdrk-proto` (M39.1 Tests 1): round trips, the bit stuffing corner cases
//! `0xFF`, `0x7E` and `0x7C`, abort, bad FCS with and without `emit_bad`, and frames the
//! decoder must drop.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use sdrk_proto::crc::crc16_x25;
use sdrk_proto::hdlc::{self, Decoder, Encoder, FLAG, MAX_PAYLOAD, MIN_PAYLOAD};
use sdrk_proto::{Error, Packet, PacketFlags, nrzi};

/// The bits of `bytes`, least significant bit first.
fn lsb_bits(bytes: &[u8]) -> Vec<u8> {
    bytes
        .iter()
        .flat_map(|&b| (0..8).map(move |k| (b >> k) & 1))
        .collect()
}

/// An HDLC frame built independently of `hdlc::Encoder`: `pre` flags, the stuffed bits of
/// `body` (sent as they are: the caller appends any FCS), `tail` flags.
fn reference_frame(body: &[u8], pre: usize, tail: usize) -> Vec<u8> {
    let flag = lsb_bits(&[FLAG]);
    let mut out = Vec::new();
    for _ in 0..pre {
        out.extend_from_slice(&flag);
    }
    let mut ones = 0;
    for b in lsb_bits(body) {
        out.push(b);
        if b == 1 {
            ones += 1;
            if ones == 5 {
                out.push(0);
                ones = 0;
            }
        } else {
            ones = 0;
        }
    }
    for _ in 0..tail {
        out.extend_from_slice(&flag);
    }
    out
}

/// Payload followed by its FCS, low byte first.
fn with_fcs(payload: &[u8]) -> Vec<u8> {
    let mut v = payload.to_vec();
    v.extend_from_slice(&crc16_x25(payload).to_le_bytes());
    v
}

fn decode_all(dec: &mut Decoder, bits: &[u8], first_index: u64) -> Vec<Packet> {
    bits.iter()
        .enumerate()
        .filter_map(|(i, &b)| dec.push_bit(b != 0, first_index + i as u64))
        .collect()
}

fn encode_vec(p: &Packet, pre: usize, tail: usize) -> Vec<u8> {
    Encoder::new(p, pre, tail).unwrap().collect()
}

/// A small deterministic byte source for the payloads.
fn lcg_bytes(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 24) as u8
        })
        .collect()
}

#[test]
fn nrzi_zero_toggles_one_keeps() {
    let bits = [true, false, false, true, false, true, true];
    let mut level = false;
    let levels: Vec<bool> = bits
        .iter()
        .map(|&b| {
            level = nrzi::encode(level, b);
            level
        })
        .collect();
    assert_eq!(levels, [false, true, false, false, true, true, true]);
    let mut dec = nrzi::Decoder::new(false);
    let back: Vec<bool> = levels.iter().map(|&l| dec.decode(l)).collect();
    assert_eq!(back, bits);
    assert!(dec.level());
    dec.reset(false);
    assert!(!dec.level());
    assert_eq!(nrzi::Decoder::default(), nrzi::Decoder::new(false));
}

#[test]
fn nrzi_is_independent_of_the_polarity() {
    let bits: Vec<bool> = lcg_bytes(7, 200).iter().map(|b| b & 1 != 0).collect();
    let mut level = true;
    let levels: Vec<bool> = bits
        .iter()
        .map(|&b| {
            level = nrzi::encode(level, b);
            level
        })
        .collect();
    let mut dec = nrzi::Decoder::new(true);
    let mut inv = nrzi::Decoder::new(false);
    for (&l, &b) in levels.iter().zip(&bits) {
        assert_eq!(dec.decode(l), b);
        assert_eq!(inv.decode(!l), b);
    }
}

#[test]
fn encoder_matches_the_reference_stuffing() {
    for payload in [
        vec![0xFF; 40],
        vec![0x7E; 40],
        vec![0x7C; 40],
        vec![0x00, 0x00],
        vec![0x3E, 0x1F, 0xF8, 0x7C],
        lcg_bytes(1, 256),
    ] {
        let p = Packet::new(0, PacketFlags::NONE, &payload).unwrap();
        let enc = Encoder::new(&p, 3, 2).unwrap();
        assert_eq!(enc.fcs(), crc16_x25(&payload));
        let bits: Vec<u8> = enc.collect();
        assert_eq!(bits, reference_frame(&with_fcs(&payload), 3, 2));
        assert!(bits.len() <= hdlc::max_bits(payload.len(), 3, 2));
    }
}

#[test]
fn stuffed_bits_of_ff_7e_and_7c() {
    let flag = [0u8, 1, 1, 1, 1, 1, 1, 0];
    // 0xFF 0xFF: five 1s, a stuffed 0, five 1s, a stuffed 0, ...
    let p = Packet::new(0, PacketFlags::NONE, &[0xFF, 0xFF]).unwrap();
    let bits = encode_vec(&p, 1, 0);
    assert_eq!(&bits[..8], &flag);
    assert_eq!(
        &bits[8..27],
        &[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1]
    );
    // 0x7E inside a frame: LSB first 0 1 1 1 1 1 [0] 1 0: never a flag.
    let p = Packet::new(0, PacketFlags::NONE, &[0x7E, 0x00]).unwrap();
    let bits = encode_vec(&p, 1, 0);
    assert_eq!(&bits[8..17], &[0, 1, 1, 1, 1, 1, 0, 1, 0]);
    // 0x7C: LSB first 0 0 1 1 1 1 1 0: five 1s are followed by a stuffed 0, then the data 0.
    let p = Packet::new(0, PacketFlags::NONE, &[0x7C, 0x00]).unwrap();
    let bits = encode_vec(&p, 1, 0);
    assert_eq!(&bits[8..17], &[0, 0, 1, 1, 1, 1, 1, 0, 0]);
}

#[test]
fn no_six_ones_between_the_flags() {
    let p = Packet::new(0, PacketFlags::NONE, &[0xFF; MAX_PAYLOAD]).unwrap();
    let bits = encode_vec(&p, 1, 1);
    let body = &bits[8..bits.len() - 8];
    assert!(body.windows(6).all(|w| w != [1, 1, 1, 1, 1, 1]));
    // 256 bytes of 0xFF and the FCS: one stuffed bit per five 1s of the payload at least.
    assert!(bits.len() >= 16 + 8 * 258 + 2048 / 5);
}

#[test]
fn round_trip_corner_cases_and_random_payloads() {
    let mut payloads = vec![
        vec![0xFF; MAX_PAYLOAD],
        vec![0x7E; MAX_PAYLOAD],
        vec![0x7C; MAX_PAYLOAD],
        vec![0xFF, 0x7E],
        vec![0x7C, 0x7E, 0xFF, 0x00, 0x01, 0x80],
        vec![0x00; MIN_PAYLOAD],
    ];
    for k in 0..50u32 {
        let len = MIN_PAYLOAD + (k as usize * 37) % (MAX_PAYLOAD - MIN_PAYLOAD + 1);
        payloads.push(lcg_bytes(100 + k, len));
    }
    for payload in payloads {
        let p = Packet::new(0, PacketFlags::NONE, &payload).unwrap();
        let bits = encode_vec(&p, 4, 2);
        let mut dec = Decoder::new(false);
        let got = decode_all(&mut dec, &bits, 1000);
        assert_eq!(got.len(), 1, "payload {payload:02x?}");
        assert_eq!(got[0].data(), payload.as_slice());
        assert_eq!(got[0].flags, PacketFlags::CRC_OK);
        // The first bit after the last opening flag.
        assert_eq!(got[0].index, 1000 + 4 * 8);
        // Through NRZI too, starting from either level.
        for start in [false, true] {
            let mut level = start;
            let mut nd = nrzi::Decoder::new(!start);
            let mut dec = Decoder::new(false);
            let mut got = Vec::new();
            for (i, &b) in bits.iter().enumerate() {
                level = nrzi::encode(level, b != 0);
                if let Some(packet) = dec.push_bit(nd.decode(level), i as u64) {
                    got.push(packet);
                }
            }
            assert_eq!(got.len(), 1);
            assert_eq!(got[0].data(), payload.as_slice());
        }
    }
}

#[test]
fn back_to_back_frames_share_a_flag_and_idle_flags_give_nothing() {
    let a = Packet::new(0, PacketFlags::NONE, b"first").unwrap();
    let b = Packet::new(0, PacketFlags::NONE, b"second").unwrap();
    let mut bits = lsb_bits(&[FLAG; 10]);
    bits.extend(encode_vec(&a, 1, 1));
    let b_start = bits.len();
    bits.extend(encode_vec(&b, 0, 1)); // the closing flag of `a` opens `b`
    bits.extend(lsb_bits(&[FLAG; 10]));
    let mut dec = Decoder::new(true);
    let got = decode_all(&mut dec, &bits, 0);
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].data(), b"first");
    assert_eq!(got[0].index, 88);
    assert_eq!(got[1].data(), b"second");
    assert_eq!(got[1].index, b_start as u64);
    assert!(!dec.is_hunting());
    dec.reset();
    assert!(dec.is_hunting());
}

#[test]
fn bad_fcs_only_with_emit_bad() {
    let payload = [0u8; 10];
    let p = Packet::new(0, PacketFlags::NONE, &payload).unwrap();
    let mut bits = encode_vec(&p, 1, 1);
    // Bit 3 of the first payload byte (no stuffing in a run of zeros).
    bits[8 + 3] = 1;
    assert_eq!(decode_all(&mut Decoder::new(false), &bits, 0), []);
    let got = decode_all(&mut Decoder::new(true), &bits, 0);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].flags, PacketFlags::CRC_FAIL);
    let mut expected = payload;
    expected[0] = 0x08;
    assert_eq!(got[0].data(), &expected);
    assert!(Decoder::new(true).emit_bad());
    assert!(!Decoder::default().emit_bad());
}

#[test]
fn seven_ones_abort_the_frame() {
    let p = Packet::new(0, PacketFlags::NONE, b"abort me").unwrap();
    let good = Packet::new(0, PacketFlags::NONE, b"next").unwrap();
    let mut bits = encode_vec(&p, 1, 0);
    let cut = bits.len() - 20;
    bits.truncate(cut);
    bits.extend_from_slice(&[1; 7]);
    // The rest of the aborted frame and its flag: the flag must not close it.
    let rest = encode_vec(&p, 1, 1);
    bits.extend_from_slice(&rest[rest.len() - 28..]);
    let after_abort = bits.len();
    bits.extend(encode_vec(&good, 1, 1));
    for emit_bad in [false, true] {
        let got = decode_all(&mut Decoder::new(emit_bad), &bits, 0);
        assert_eq!(got.len(), 1, "emit_bad {emit_bad}");
        assert_eq!(got[0].data(), b"next");
        assert_eq!(got[0].index, after_abort as u64 + 8);
    }
    // An abort right after the opening flag, then idle 1s: nothing.
    let mut bits = lsb_bits(&[FLAG]);
    bits.extend_from_slice(&[1; 40]);
    assert_eq!(decode_all(&mut Decoder::new(true), &bits, 0), []);
}

#[test]
fn frames_the_decoder_drops() {
    let mut dec = Decoder::new(true);
    // Between flags: 3 bytes (shorter than payload and FCS of 2 + 2 bytes).
    let short = reference_frame(&with_fcs(&[0x42]), 1, 1);
    assert_eq!(decode_all(&mut dec, &short, 0), []);
    // 13 bits: not whole bytes.
    let mut bits = lsb_bits(&[FLAG]);
    bits.extend_from_slice(&[0; 13]);
    bits.extend(lsb_bits(&[FLAG]));
    assert_eq!(decode_all(&mut dec, &bits, 0), []);
    // 257 payload bytes and the FCS: too long; the next frame still decodes.
    let long = lcg_bytes(5, MAX_PAYLOAD + 1);
    let mut bits = reference_frame(&with_fcs(&long), 1, 1);
    let ok = Packet::new(0, PacketFlags::NONE, &long[..MAX_PAYLOAD]).unwrap();
    bits.extend(encode_vec(&ok, 1, 1));
    let got = decode_all(&mut dec, &bits, 0);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].data(), &long[..MAX_PAYLOAD]);
    assert_eq!(got[0].flags, PacketFlags::CRC_OK);
}

#[test]
fn encoder_rejects_short_payloads_and_encode_checks_the_buffer() {
    for len in 0..MIN_PAYLOAD {
        let p = Packet::new(0, PacketFlags::NONE, &vec![0; len]).unwrap();
        assert!(matches!(
            Encoder::new(&p, 1, 1),
            Err(Error::InvalidArgument(_))
        ));
    }
    let p = Packet::new(0, PacketFlags::NONE, &[0xFF; 4]).unwrap();
    assert!(Encoder::new(&p, usize::MAX, 1).is_err());
    let n = encode_vec(&p, 2, 2).len();
    let mut out = vec![0u8; n];
    assert_eq!(hdlc::encode(&p, 2, 2, &mut out).unwrap(), n);
    assert_eq!(
        hdlc::encode(&p, 2, 2, &mut out[..n - 1]),
        Err(Error::BufferTooSmall {
            needed: n,
            got: n - 1
        })
    );
}

#[test]
fn size_hint_brackets_the_remaining_bits() {
    let p = Packet::new(0, PacketFlags::NONE, &[0xFF, 0x7E, 0x7C, 0x00, 0xFF]).unwrap();
    let mut enc = Encoder::new(&p, 2, 3).unwrap();
    let total = enc.clone().count();
    for done in 0..=total {
        let (low, high) = enc.size_hint();
        let left = total - done;
        assert!(low <= left && left <= high.unwrap(), "at {done}");
        if done < total {
            assert!(!enc.is_done());
        }
        enc.next();
    }
    assert!(enc.is_done());
    assert_eq!(enc.size_hint(), (0, Some(0)));
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Encoder>();
    assert_send_sync::<Decoder>();
    assert_send_sync::<nrzi::Decoder>();
}
