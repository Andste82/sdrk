//! PN9 whitening (M39 §5): the sequence of `sdrk_proto::whiten::Pn9` and the option
//! `Whitening` of the test framing (round trip, the layout of a whitened frame, `None`
//! unchanged). The payloads are fixed patterns; no random data.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::assert_is_empty
)]

use sdrk_proto::crc::crc16_ccitt_false;
use sdrk_proto::frame::{
    self, Decoder, FrameBits, MAX_PAYLOAD, PREAMBLE_BYTE, SYNC_WORD, Whitening,
};
use sdrk_proto::whiten::{PN9_PERIOD, PN9_SEED, Pn9, pn9_byte};
use sdrk_proto::{Packet, PacketFlags};

/// A payload of `len` bytes that differs for every length.
fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 37 + len * 11 + 5) as u8).collect()
}

fn packet(payload: &[u8]) -> Packet {
    Packet::new(0, PacketFlags::NONE, payload).unwrap()
}

fn frame_bytes(p: &Packet, preamble: usize, w: Whitening) -> Vec<u8> {
    let mut out = vec![0u8; frame::frame_bits(p.len(), preamble) / 8];
    let n = frame::encode_bytes(p, preamble, w, &mut out).unwrap();
    assert_eq!(n, out.len());
    out
}

fn frame_bits(p: &Packet, preamble: usize, w: Whitening) -> Vec<u8> {
    let mut out = vec![0u8; frame::frame_bits(p.len(), preamble)];
    let n = frame::encode(p, preamble, w, &mut out).unwrap();
    assert_eq!(n, out.len());
    out
}

fn unpack(bytes: &[u8]) -> Vec<u8> {
    bytes
        .iter()
        .flat_map(|b| (0..8).rev().map(move |k| (b >> k) & 1))
        .collect()
}

fn decode(dec: &mut Decoder, bits: &[u8]) -> Vec<Packet> {
    bits.iter()
        .enumerate()
        .filter_map(|(i, &b)| dec.push_bit(b != 0, i as u64))
        .collect()
}

#[test]
fn the_first_bytes_of_pn9() {
    let mut pn9 = Pn9::new();
    let first: Vec<u8> = (0..8).map(|_| pn9.next_byte()).collect();
    assert_eq!(first, [0xFF, 0xE1, 0x1D, 0x9A, 0xED, 0x85, 0x33, 0x24]);
    assert_eq!(PN9_SEED, 0x1FF);
}

#[test]
fn pn9_is_the_register_of_x9_x5_1() {
    // Bit-serial model: the low byte of the 9-bit register, then 8 shifts right with
    // bit0 ^ bit5 entering at bit 8.
    let mut state: u16 = 0x1FF;
    let mut pn9 = Pn9::new();
    for k in 0..3 * PN9_PERIOD {
        assert_eq!(pn9.next_byte(), (state & 0xFF) as u8, "byte {k}");
        assert_eq!(pn9_byte(k), (state & 0xFF) as u8, "byte {k}");
        for _ in 0..8 {
            let bit = (state ^ (state >> 5)) & 1;
            state = (state >> 1) | (bit << 8);
        }
    }
    // The bit stream (bytes least significant bit first) obeys b[t + 9] = b[t] ^ b[t + 5].
    let mut pn9 = Pn9::new();
    let bits: Vec<u8> = (0..PN9_PERIOD + 4)
        .map(|_| pn9.next_byte())
        .flat_map(|b| (0..8).map(move |i| (b >> i) & 1))
        .collect();
    for t in 0..bits.len() - 9 {
        assert_eq!(bits[t + 9], bits[t] ^ bits[t + 5], "bit {t}");
    }
}

#[test]
fn pn9_has_period_511_with_balanced_bits() {
    let seq: Vec<u8> = (0..PN9_PERIOD).map(pn9_byte).collect();
    for p in 1..PN9_PERIOD {
        assert!(
            (0..PN9_PERIOD).any(|k| seq[k] != seq[(k + p) % PN9_PERIOD]),
            "period {p}"
        );
    }
    // A maximal-length sequence: 8 * 256 ones in 8 periods of 511 bits.
    let ones: u32 = seq.iter().map(|b| b.count_ones()).sum();
    assert_eq!(ones, 8 * 256);
    let mut pn9 = Pn9::new();
    for _ in 0..PN9_PERIOD {
        pn9.next_byte();
    }
    assert_eq!(pn9.position(), 0);
    assert_eq!(pn9.next_byte(), 0xFF);
}

#[test]
fn apply_in_pieces_equals_apply_at_once() {
    let data: Vec<u8> = pattern(1500);
    let mut whole = data.clone();
    Pn9::new().apply(&mut whole);
    for (k, (w, d)) in whole.iter().zip(&data).enumerate() {
        assert_eq!(*w, d ^ pn9_byte(k));
    }
    for piece in [1usize, 7, 255, 511, 512, 1000] {
        let mut pn9 = Pn9::new();
        let mut parts = data.clone();
        for chunk in parts.chunks_mut(piece) {
            pn9.apply(chunk);
        }
        assert_eq!(parts, whole, "piece {piece}");
        assert_eq!(pn9.position(), 1500 % PN9_PERIOD);
    }
    // Twice restores the data.
    Pn9::new().apply(&mut whole);
    assert_eq!(whole, data);
    let mut pn9 = Pn9::new();
    pn9.apply(&mut []);
    assert_eq!(pn9, Pn9::default());
}

#[test]
fn a_whitened_frame_xors_length_payload_and_crc() {
    for len in [0usize, 1, 2, 17, 252, MAX_PAYLOAD] {
        for preamble in [0usize, 1, 4, 32] {
            let payload = pattern(len);
            let p = packet(&payload);
            let plain = frame_bytes(&p, preamble, Whitening::None);
            let white = frame_bytes(&p, preamble, Whitening::Pn9);
            assert_eq!(plain.len(), white.len());
            // Preamble and sync word are not whitened.
            let head = preamble + 4;
            assert_eq!(white[..head], plain[..head]);
            assert!(white[..preamble].iter().all(|&b| b == PREAMBLE_BYTE));
            assert_eq!(white[preamble..head], SYNC_WORD.to_be_bytes());
            // Length, payload and CRC (over the plain length and payload) are XORed with PN9.
            let mut body = vec![len as u8];
            body.extend_from_slice(&payload);
            let crc = crc16_ccitt_false(&body);
            body.extend_from_slice(&crc.to_be_bytes());
            assert_eq!(plain[head..], body);
            for (k, (w, b)) in white[head..].iter().zip(&body).enumerate() {
                assert_eq!(
                    *w,
                    b ^ pn9_byte(k),
                    "len {len} preamble {preamble} byte {k}"
                );
            }
            // The bit forms agree with the packed bytes.
            assert_eq!(frame_bits(&p, preamble, Whitening::Pn9), unpack(&white));
            let iter: Vec<u8> = FrameBits::new(&p, preamble, Whitening::Pn9)
                .unwrap()
                .collect();
            assert_eq!(iter, unpack(&white));
        }
    }
}

#[test]
fn whitening_none_leaves_the_frame_unchanged() {
    // `Whitening::None` is the M37 framing: the bytes of the documented layout.
    let p = packet(&[0xA5, 0x00]);
    let bytes = frame_bytes(&p, 2, Whitening::None);
    let crc = crc16_ccitt_false(&[2, 0xA5, 0x00]).to_be_bytes();
    assert_eq!(
        bytes,
        [
            0x55, 0x55, 0x1A, 0xCF, 0xFC, 0x1D, 2, 0xA5, 0x00, crc[0], crc[1]
        ]
    );
    assert_eq!(Whitening::default(), Whitening::None);
    let dec = Decoder::new(0, false, Whitening::default()).unwrap();
    assert_eq!(dec.whitening(), Whitening::None);
}

#[test]
fn round_trip_with_whitening() {
    // Every payload length, back to back, with a whitened all-zero payload among them.
    let mut stream = Vec::new();
    let mut sent = Vec::new();
    for len in 0..=MAX_PAYLOAD {
        let payload = if len == 100 {
            vec![0u8; len]
        } else {
            pattern(len)
        };
        let p = packet(&payload);
        stream.extend(frame_bits(&p, 4, Whitening::Pn9));
        sent.push(payload);
    }
    let mut dec = Decoder::new(0, false, Whitening::Pn9).unwrap();
    assert_eq!(dec.whitening(), Whitening::Pn9);
    let got = decode(&mut dec, &stream);
    assert_eq!(got.len(), sent.len());
    for (g, s) in got.iter().zip(&sent) {
        assert_eq!(g.data(), s.as_slice());
        assert_eq!(g.flags, PacketFlags::CRC_OK);
    }
}

#[test]
fn whitening_must_match_on_both_sides() {
    let p = packet(&pattern(40));
    let white = frame_bits(&p, 4, Whitening::Pn9);
    let plain = frame_bits(&p, 4, Whitening::None);
    // A decoder without whitening reads a wrong length or a wrong CRC.
    let got = decode(&mut Decoder::new(0, true, Whitening::None).unwrap(), &white);
    assert!(got.iter().all(|g| g.flags == PacketFlags::CRC_FAIL));
    let got = decode(&mut Decoder::new(0, true, Whitening::Pn9).unwrap(), &plain);
    assert!(got.iter().all(|g| g.flags == PacketFlags::CRC_FAIL));
    assert!(
        decode(
            &mut Decoder::new(0, false, Whitening::None).unwrap(),
            &white
        )
        .is_empty()
    );
    assert!(decode(&mut Decoder::new(0, false, Whitening::Pn9).unwrap(), &plain).is_empty());
}

#[test]
fn whitening_breaks_up_long_runs() {
    // An all-zero payload of 253 bytes is a run of 2024 equal bits without whitening; with
    // PN9 the longest run after the sync word is at most two bytes long.
    let p = packet(&[0u8; MAX_PAYLOAD]);
    let longest = |bits: &[u8]| {
        let (mut best, mut run) = (0usize, 0usize);
        for (i, b) in bits.iter().enumerate() {
            run = if i > 0 && bits[i - 1] == *b {
                run + 1
            } else {
                1
            };
            best = best.max(run);
        }
        best
    };
    let after_sync = 8 * (4 + 4);
    let plain = frame_bits(&p, 4, Whitening::None);
    let white = frame_bits(&p, 4, Whitening::Pn9);
    assert!(longest(&plain[after_sync..]) >= 8 * MAX_PAYLOAD);
    assert!(longest(&white[after_sync..]) <= 16);
}
