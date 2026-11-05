//! The test framing of `sdrk-proto` (M37 §3, Tests 1): round trip of every length, sync search
//! after arbitrary bits and with sync-word errors, the property "a corrupted frame never gives
//! `CRC_OK` unless the corruption is in the preamble", and the golden frames of
//! `tests/golden/generate.py`. Random data uses seed 479.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use sdrk_proto::frame::{
    self, Decoder, FrameBits, MAX_PAYLOAD, MAX_SYNC_ERRORS, PREAMBLE_BYTE, SYNC_WORD, Whitening,
};
use sdrk_proto::{Error, Packet, PacketFlags};
use sdrk_testkit::Pcg32;
use sdrk_testkit::golden::{self, Array};

const SEED: u64 = 479;
const NONE: Vec<Packet> = Vec::new();

fn bytes(rng: &mut Pcg32, n: usize) -> Vec<u8> {
    (0..n).map(|_| rng.next_u32() as u8).collect()
}

fn encode(payload: &[u8], preamble: usize) -> Vec<u8> {
    let p = Packet::new(0, PacketFlags::NONE, payload).unwrap();
    let mut out = vec![0u8; frame::frame_bits(payload.len(), preamble)];
    let n = frame::encode(&p, preamble, Whitening::None, &mut out).unwrap();
    assert_eq!(n, out.len());
    out
}

/// Feeds `bits` (starting at index `first`) and collects every returned packet.
fn decode(dec: &mut Decoder, bits: &[u8], first: u64) -> Vec<Packet> {
    bits.iter()
        .enumerate()
        .filter_map(|(i, &b)| dec.push_bit(b != 0, first + i as u64))
        .collect()
}

#[test]
fn layout_of_a_frame() {
    let bits = encode(&[0xA5, 0x00], 2);
    assert_eq!(bits.len(), 8 * (2 + 4 + 1 + 2 + 2));
    let byte = |k: usize| {
        bits[k * 8..k * 8 + 8]
            .iter()
            .fold(0u8, |a, &b| (a << 1) | b)
    };
    assert_eq!(byte(0), PREAMBLE_BYTE);
    assert_eq!(byte(1), PREAMBLE_BYTE);
    let sync = u32::from_be_bytes([byte(2), byte(3), byte(4), byte(5)]);
    assert_eq!(sync, SYNC_WORD);
    assert_eq!(byte(6), 2);
    assert_eq!((byte(7), byte(8)), (0xA5, 0x00));
    let crc = sdrk_proto::crc::crc16_ccitt_false(&[2, 0xA5, 0x00]);
    assert_eq!(u16::from_be_bytes([byte(9), byte(10)]), crc);
}

#[test]
fn round_trip_every_length() {
    let mut rng = Pcg32::new(SEED, 1);
    let mut dec = Decoder::new(0, false, Whitening::None).unwrap();
    for len in 0..=MAX_PAYLOAD {
        let payload = bytes(&mut rng, len);
        let bits = encode(&payload, 4);
        let got = decode(&mut dec, &bits, 1000);
        assert_eq!(got.len(), 1, "length {len}");
        assert_eq!(got[0].data(), payload.as_slice(), "length {len}");
        assert_eq!(got[0].flags, PacketFlags::CRC_OK);
        assert_eq!(got[0].index, 1000 + 32, "length {len}");
        assert!(dec.is_hunting());
    }
}

#[test]
fn back_to_back_frames_without_preamble() {
    let mut rng = Pcg32::new(SEED, 2);
    let payloads: Vec<Vec<u8>> = (0..20)
        .map(|_| {
            let n = rng.below(40) as usize;
            bytes(&mut rng, n)
        })
        .collect();
    let mut stream = Vec::new();
    let mut starts = Vec::new();
    for p in &payloads {
        starts.push(stream.len() as u64);
        stream.extend(encode(p, 0));
    }
    let got = decode(
        &mut Decoder::new(0, false, Whitening::None).unwrap(),
        &stream,
        0,
    );
    assert_eq!(got.len(), payloads.len());
    for ((g, p), s) in got.iter().zip(&payloads).zip(&starts) {
        assert_eq!(g.data(), p.as_slice());
        assert_eq!(g.index, *s);
    }
}

#[test]
fn finds_a_frame_after_arbitrary_leading_bits() {
    let mut rng = Pcg32::new(SEED, 3);
    for trial in 0..200 {
        let lead = rng.below(600) as usize;
        let leading: Vec<u8> = (0..lead).map(|_| (rng.next_u32() & 1) as u8).collect();
        let len = rng.below(MAX_PAYLOAD as u32 + 1) as usize;
        let payload = bytes(&mut rng, len);
        let preamble = rng.below(5) as usize;
        let mut stream = leading;
        stream.extend(encode(&payload, preamble));
        let got = decode(
            &mut Decoder::new(0, false, Whitening::None).unwrap(),
            &stream,
            7,
        );
        assert_eq!(got.len(), 1, "trial {trial}");
        assert_eq!(got[0].data(), payload.as_slice(), "trial {trial}");
        assert_eq!(
            got[0].index,
            7 + (lead + preamble * 8) as u64,
            "trial {trial}"
        );
    }
}

#[test]
fn accepts_up_to_max_sync_errors() {
    let mut rng = Pcg32::new(SEED, 4);
    let payload = bytes(&mut rng, 17);
    let clean = encode(&payload, 4);
    let sync_start = 32;
    for max in 0..=MAX_SYNC_ERRORS {
        for errors in 0..=MAX_SYNC_ERRORS {
            for _ in 0..20 {
                let mut bits = clean.clone();
                // `errors` distinct positions in the sync word.
                let mut pos: Vec<usize> = Vec::new();
                while pos.len() < errors as usize {
                    let p = rng.below(32) as usize;
                    if !pos.contains(&p) {
                        pos.push(p);
                    }
                }
                for p in &pos {
                    bits[sync_start + p] ^= 1;
                }
                let got = decode(
                    &mut Decoder::new(max, false, Whitening::None).unwrap(),
                    &bits,
                    0,
                );
                if errors <= max {
                    assert_eq!(got.len(), 1, "max {max}, errors {pos:?}");
                    assert_eq!(got[0].data(), payload.as_slice());
                    assert_eq!(got[0].index, 32);
                    assert_eq!(got[0].flags, PacketFlags::CRC_OK);
                } else {
                    assert_eq!(got, NONE, "max {max}, errors {pos:?}");
                }
            }
        }
    }
    assert!(matches!(
        Decoder::new(MAX_SYNC_ERRORS + 1, false, Whitening::None),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn bad_crc_is_delivered_only_with_emit_bad() {
    let payload = b"payload".to_vec();
    let mut bits = encode(&payload, 4);
    // Flip the last bit of the CRC.
    let last = bits.len() - 1;
    bits[last] ^= 1;
    assert_eq!(
        decode(
            &mut Decoder::new(0, false, Whitening::None).unwrap(),
            &bits,
            0
        ),
        NONE
    );
    let got = decode(
        &mut Decoder::new(0, true, Whitening::None).unwrap(),
        &bits,
        0,
    );
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].flags, PacketFlags::CRC_FAIL);
    assert_eq!(got[0].data(), payload.as_slice());
    // A payload bit error: the payload is returned as received.
    let mut bits = encode(&payload, 4);
    bits[32 + 32 + 8] ^= 1;
    let got = decode(
        &mut Decoder::new(0, true, Whitening::None).unwrap(),
        &bits,
        0,
    );
    assert_eq!(got[0].flags, PacketFlags::CRC_FAIL);
    assert_eq!(got[0].data()[0], payload[0] ^ 0x80);
}

#[test]
fn length_above_max_payload_ends_the_frame() {
    // A sync word followed by the length 254 and then a valid frame: the bogus frame is
    // dropped after its length byte and the hunt finds the next frame.
    let mut bits: Vec<u8> = (0..32)
        .map(|i| ((SYNC_WORD >> (31 - i)) & 1) as u8)
        .collect();
    bits.extend((0..8).map(|i| (254u8 >> (7 - i)) & 1));
    let start = bits.len() as u64;
    bits.extend(encode(b"next", 0));
    let mut dec = Decoder::new(0, true, Whitening::None).unwrap();
    let got = decode(&mut dec, &bits, 0);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].data(), b"next");
    assert_eq!(got[0].index, start);
}

#[test]
fn reset_drops_a_partial_frame() {
    let bits = encode(b"abcdef", 4);
    let mut dec = Decoder::new(0, false, Whitening::None).unwrap();
    assert_eq!(decode(&mut dec, &bits[..bits.len() / 2], 0), NONE);
    assert!(!dec.is_hunting());
    dec.reset();
    assert!(dec.is_hunting());
    assert_eq!(decode(&mut dec, &bits[bits.len() / 2..], 0), NONE);
    assert_eq!(decode(&mut dec, &bits, 0).len(), 1);
}

#[test]
fn encode_errors() {
    let long = Packet::new(0, PacketFlags::NONE, &[0u8; MAX_PAYLOAD + 1]).unwrap();
    let mut out = vec![0u8; 4000];
    assert_eq!(
        frame::encode(&long, 4, Whitening::None, &mut out),
        Err(Error::TooLong {
            len: MAX_PAYLOAD + 1,
            max: MAX_PAYLOAD
        })
    );
    assert!(FrameBits::new(&long, 0, Whitening::None).is_err());
    let p = Packet::new(0, PacketFlags::NONE, b"x").unwrap();
    let need = frame::frame_bits(1, 4);
    assert_eq!(
        frame::encode(&p, 4, Whitening::None, &mut out[..need - 1]),
        Err(Error::BufferTooSmall {
            needed: need,
            got: need - 1
        })
    );
    assert!(FrameBits::new(&p, usize::MAX, Whitening::None).is_err());
    assert_eq!(frame::frame_bits(1, usize::MAX), usize::MAX);
}

#[test]
fn frame_bits_resume_anywhere() {
    let mut rng = Pcg32::new(SEED, 5);
    let payload = bytes(&mut rng, 100);
    let p = Packet::new(0, PacketFlags::NONE, &payload).unwrap();
    let whole = encode(&payload, 3);
    let mut it = FrameBits::new(&p, 3, Whitening::None).unwrap();
    assert_eq!(it.len(), whole.len());
    let mut got = Vec::new();
    while it.len() > 0 {
        let take = 1 + rng.below(50) as usize;
        let saved = it.clone(); // an encoder keeps the iterator between calls
        got.extend(saved.take(take));
        for _ in 0..take.min(it.len()) {
            it.next();
        }
        assert_eq!(it.position(), got.len());
    }
    assert_eq!(got, whole);
    assert_eq!(it.next(), None);
}

/// Feeds the frame with 1 to 3 flipped bits, followed by random bits, and returns the packets.
fn corrupted(
    rng: &mut Pcg32,
    payload: &[u8],
    preamble: usize,
    region: (usize, usize),
) -> (Vec<Packet>, Vec<usize>) {
    let mut bits = encode(payload, preamble);
    let errors = 1 + rng.below(3) as usize;
    let mut pos: Vec<usize> = Vec::new();
    while pos.len() < errors {
        let p = region.0 + rng.below((region.1 - region.0) as u32) as usize;
        if !pos.contains(&p) {
            pos.push(p);
        }
    }
    for &p in &pos {
        bits[p] ^= 1;
    }
    // Random bits after the frame, so a length error that makes the frame longer is read to
    // its (wrong) end.
    bits.extend((0..2200).map(|_| (rng.next_u32() & 1) as u8));
    (
        decode(
            &mut Decoder::new(0, true, Whitening::None).unwrap(),
            &bits,
            0,
        ),
        pos,
    )
}

#[test]
fn corrupted_frames_never_give_crc_ok() {
    let mut rng = Pcg32::new(SEED, 6);
    for trial in 0..3000 {
        let len = rng.below(MAX_PAYLOAD as u32 + 1) as usize;
        let payload = bytes(&mut rng, len);
        let preamble = rng.below(5) as usize;
        let total = frame::frame_bits(len, preamble);
        // Errors anywhere after the preamble: sync word, length, payload or CRC.
        let (got, pos) = corrupted(&mut rng, &payload, preamble, (preamble * 8, total));
        for g in &got {
            // A random tail may contain a frame by chance; it never carries our payload.
            assert!(
                !(g.flags.contains(PacketFlags::CRC_OK) && g.index < total as u64),
                "trial {trial}: errors at {pos:?} gave {g:?}"
            );
        }
    }
}

#[test]
fn errors_in_payload_and_crc_are_always_detected() {
    // CRC-16/CCITT has Hamming distance 4 for these lengths: up to 3 errors in length (when it
    // stays the same frame), payload and CRC are always detected; here only payload and CRC.
    let mut rng = Pcg32::new(SEED, 7);
    for trial in 0..3000 {
        let len = rng.below(MAX_PAYLOAD as u32 + 1) as usize;
        let payload = bytes(&mut rng, len);
        let total = frame::frame_bits(len, 0);
        let (got, pos) = corrupted(&mut rng, &payload, 0, (40, total));
        assert!(
            !got.is_empty(),
            "trial {trial}: emit_bad must deliver the frame"
        );
        assert_eq!(
            got[0].flags,
            PacketFlags::CRC_FAIL,
            "trial {trial}: {pos:?}"
        );
        assert_eq!(got[0].len(), len);
    }
}

#[test]
fn errors_in_the_preamble_do_not_matter() {
    let mut rng = Pcg32::new(SEED, 8);
    for trial in 0..500 {
        let len = rng.below(MAX_PAYLOAD as u32 + 1) as usize;
        let payload = bytes(&mut rng, len);
        let preamble = 1 + rng.below(8) as usize;
        let (got, pos) = corrupted(&mut rng, &payload, preamble, (0, preamble * 8));
        assert_eq!(got[0].flags, PacketFlags::CRC_OK, "trial {trial}: {pos:?}");
        assert_eq!(got[0].data(), payload.as_slice());
        assert_eq!(got[0].index, (preamble * 8) as u64);
    }
}

fn i32s(a: &Array) -> Vec<i32> {
    match a {
        Array::I32(v) => v.clone(),
        other => panic!("expected i32, got {}", other.dtype()),
    }
}

fn u8s(a: &Array) -> Vec<u8> {
    match a {
        Array::U8(v) => v.clone(),
        other => panic!("expected u8, got {}", other.dtype()),
    }
}

#[test]
fn golden_frame_encode() {
    let case = golden::load("proto", "frame_encode").unwrap();
    let payload = u8s(case.array("payload").unwrap());
    let lengths = i32s(case.array("lengths").unwrap());
    let preambles = i32s(case.array("preamble_bytes").unwrap());
    let want = u8s(case.array("bits").unwrap());
    assert_eq!(case.param_u64("frames").unwrap() as usize, lengths.len());
    let (mut pos, mut got) = (0usize, Vec::new());
    let mut packets = Vec::new();
    let mut starts = Vec::new();
    for (&len, &pre) in lengths.iter().zip(&preambles) {
        let (len, pre) = (len as usize, pre as usize);
        let p = Packet::new(0, PacketFlags::NONE, &payload[pos..pos + len]).unwrap();
        pos += len;
        starts.push((got.len() + pre * 8) as u64);
        let mut out = vec![0u8; frame::frame_bits(len, pre)];
        frame::encode(&p, pre, Whitening::None, &mut out).unwrap();
        got.extend(out);
        packets.push(p);
    }
    assert_eq!(pos, payload.len());
    assert_eq!(got, want, "frame bits differ from the golden model");
    // The decoder recovers every frame of the golden stream with its sync-word index.
    let dec = decode(
        &mut Decoder::new(0, false, Whitening::None).unwrap(),
        &want,
        0,
    );
    assert_eq!(dec.len(), packets.len());
    for ((d, p), s) in dec.iter().zip(&packets).zip(&starts) {
        assert_eq!(d.data(), p.data());
        assert_eq!(d.index, *s);
        assert_eq!(d.flags, PacketFlags::CRC_OK);
    }
}

#[test]
fn encode_bytes_packs_the_bits_of_encode() {
    let mut rng = Pcg32::new(SEED, 77);
    for len in [0, 1, 2, 31, 200, MAX_PAYLOAD] {
        for preamble in [0, 1, 4, 32] {
            let payload = bytes(&mut rng, len);
            let p = Packet::new(0, PacketFlags::NONE, &payload).unwrap();
            let bits = encode(&payload, preamble);
            let mut packed = vec![0u8; bits.len() / 8 + 3];
            let n = frame::encode_bytes(&p, preamble, Whitening::None, &mut packed).unwrap();
            assert_eq!(n * 8, bits.len());
            let unpacked: Vec<u8> = packed[..n]
                .iter()
                .flat_map(|b| (0..8).rev().map(move |k| (b >> k) & 1))
                .collect();
            assert_eq!(unpacked, bits, "len {len} preamble {preamble}");
        }
    }
    let p = Packet::new(0, PacketFlags::NONE, &[1, 2, 3]).unwrap();
    assert_eq!(
        frame::encode_bytes(&p, 4, Whitening::None, &mut [0u8; 13]),
        Err(Error::BufferTooSmall {
            needed: 14,
            got: 13
        })
    );
    let long = Packet::new(0, PacketFlags::NONE, &[0; 254]).unwrap();
    assert!(matches!(
        frame::encode_bytes(&long, 0, Whitening::None, &mut [0u8; 512]),
        Err(Error::TooLong { .. })
    ));
}
