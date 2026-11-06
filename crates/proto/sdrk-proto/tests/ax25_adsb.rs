//! The AX.25 address header and the Mode S fields of `sdrk-proto` (M39.1 Tests 1).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use sdrk_device::{ErrorKind, SdrkError};
use sdrk_proto::ax25::{Address, Header, MAX_DIGIPEATERS};
use sdrk_proto::crc::crc16_x25;
use sdrk_proto::{Error, Packet, PacketFlags, adsb, hdlc};

/// The APRS frame `N0CALL-7>APRS,WIDE1-1,WIDE2-1:!4903.50N/07201.75W-Test` without its FCS.
///
/// The bytes were written by hand from the address encoding of the AX.25 2.2 specification
/// (section 3.12: each callsign character shifted left by one, padded with spaces; the SSID
/// byte `CRRSSSSE` with both reserved bits set, the command bit set on the destination and
/// clear on the source of a command frame, the extension bit on the last address), the UI
/// control byte `0x03` and the PID `0xF0` (no layer 3). The information field is the position
/// report example `!4903.50N/07201.75W-` of the APRS 1.0.1 protocol reference (chapter 8)
/// followed by `Test`. Its FCS `0xAF55` was computed with a bit-by-bit CRC-16/X-25 loop
/// outside this repository.
const APRS_FRAME: [u8; 54] = [
    0x82, 0xA0, 0xA4, 0xA6, 0x40, 0x40, 0xE0, // APRS, SSID 0, C = 1
    0x9C, 0x60, 0x86, 0x82, 0x98, 0x98, 0x6E, // N0CALL, SSID 7, C = 0
    0xAE, 0x92, 0x88, 0x8A, 0x62, 0x40, 0x62, // WIDE1, SSID 1
    0xAE, 0x92, 0x88, 0x8A, 0x64, 0x40, 0x63, // WIDE2, SSID 1, last address
    0x03, 0xF0, // UI frame, no layer 3
    0x21, 0x34, 0x39, 0x30, 0x33, 0x2E, 0x35, 0x30, 0x4E, 0x2F, 0x30, 0x37, 0x32, 0x30, 0x31, 0x2E,
    0x37, 0x35, 0x57, 0x2D, 0x54, 0x65, 0x73, 0x74,
];

const MODE_S: [u8; 14] = [
    0x8D, 0x48, 0x40, 0xD6, 0x20, 0x2C, 0xC3, 0x71, 0xC3, 0x2C, 0xE0, 0x57, 0x60, 0x98,
];

#[test]
fn header_of_the_known_aprs_frame() {
    let h = Header::parse(&APRS_FRAME).unwrap();
    assert_eq!(h.to_string(), "N0CALL-7>APRS,WIDE1-1,WIDE2-1:");
    assert_eq!(h.destination().callsign(), "APRS");
    assert_eq!(h.destination().ssid(), 0);
    assert!(h.destination().ch_bit());
    assert_eq!(h.source().callsign(), "N0CALL");
    assert_eq!(h.source().ssid(), 7);
    assert!(!h.source().ch_bit());
    let digis: Vec<String> = h.digipeaters().iter().map(Address::to_string).collect();
    assert_eq!(digis, ["WIDE1-1", "WIDE2-1"]);
    assert!(!h.digipeaters()[0].ch_bit());
    assert_eq!(h.control(), 0x03);
    assert_eq!(h.pid(), Some(0xF0));
    assert_eq!(&APRS_FRAME[h.info_offset()..], b"!4903.50N/07201.75W-Test");
    assert_eq!(crc16_x25(&APRS_FRAME), 0xAF55);
    assert!(format!("{h:?}").contains("N0CALL-7"));
}

#[test]
fn the_known_aprs_frame_through_hdlc() {
    let p = Packet::new(0, PacketFlags::NONE, &APRS_FRAME).unwrap();
    let mut dec = hdlc::Decoder::new(false);
    let got: Vec<Packet> = hdlc::Encoder::new(&p, 8, 2)
        .unwrap()
        .enumerate()
        .filter_map(|(i, b)| dec.push_bit(b != 0, i as u64))
        .collect();
    assert_eq!(got.len(), 1);
    let h = Header::parse(got[0].data()).unwrap();
    assert_eq!(h.to_string(), "N0CALL-7>APRS,WIDE1-1,WIDE2-1:");
}

/// A frame from `n` digipeaters named `D0` ... with the given control byte.
fn frame_with(n: usize, control: u8) -> Vec<u8> {
    let mut f = APRS_FRAME[..14].to_vec();
    for k in 0..n {
        f.extend_from_slice(&[
            b'D' << 1,
            (b'0' + k as u8) << 1,
            0x40,
            0x40,
            0x40,
            0x40,
            0x60,
        ]);
    }
    let last = f.len() - 1;
    f[last] |= 1;
    f.push(control);
    f
}

#[test]
fn up_to_eight_digipeaters() {
    for n in 0..=MAX_DIGIPEATERS {
        let mut f = frame_with(n, 0x03);
        f.push(0xF0);
        let h = Header::parse(&f).unwrap();
        assert_eq!(h.digipeaters().len(), n);
        assert_eq!(h.info_offset(), f.len());
        if n == 2 {
            assert_eq!(h.to_string(), "N0CALL-7>APRS,D0,D1:");
        }
    }
    let mut f = frame_with(MAX_DIGIPEATERS + 1, 0x03);
    f.push(0xF0);
    let e = Header::parse(&f).unwrap_err();
    assert!(matches!(e, Error::Parse(_)));
    assert_eq!(e.kind(), ErrorKind::Parse);
    assert!(e.to_string().starts_with("parse error: "));
}

#[test]
fn pid_only_for_i_and_ui_frames() {
    // UI with the poll bit, an I frame: a PID. An S frame (RR, 0x01) and a U frame (SABM,
    // 0x2F): none.
    for (control, pid) in [
        (0x03, true),
        (0x13, true),
        (0x00, true),
        (0x01, false),
        (0x2F, false),
    ] {
        let mut f = frame_with(0, control);
        if pid {
            f.push(0xCC);
        }
        let h = Header::parse(&f).unwrap();
        assert_eq!(h.pid(), pid.then_some(0xCC), "control {control:#x}");
        assert_eq!(h.info_offset(), f.len());
    }
}

#[test]
fn malformed_headers_are_parse_errors() {
    let mut cases: Vec<Vec<u8>> = vec![
        vec![],
        APRS_FRAME[..13].to_vec(), // cut in the source
        APRS_FRAME[..28].to_vec(), // no control byte
        APRS_FRAME[..29].to_vec(), // no PID
        frame_with(0, 0x03)[..7]
            .iter()
            .copied()
            .chain([0; 7])
            .collect(), // empty source
    ];
    // Extension bit on the destination: no source.
    let mut f = APRS_FRAME.to_vec();
    f[6] |= 1;
    cases.push(f);
    // A lower-case letter, a space inside the callsign, bit 0 set in a character.
    for (at, byte) in [(7, b'n' << 1), (8, b' ' << 1), (9, (b'C' << 1) | 1)] {
        let mut f = APRS_FRAME.to_vec();
        f[at] = byte;
        cases.push(f);
    }
    for f in cases {
        assert!(
            matches!(Header::parse(&f), Err(Error::Parse(_))),
            "frame {f:02x?}"
        );
    }
}

#[test]
fn mode_s_crc_df_and_icao() {
    assert!(adsb::crc_ok(&MODE_S));
    assert_eq!(adsb::df(&MODE_S), 17);
    assert_eq!(adsb::icao(&MODE_S), 0x48_40D6);
    for bit in 0..MODE_S.len() * 8 {
        let mut f = MODE_S;
        f[bit / 8] ^= 0x80 >> (bit % 8);
        assert!(!adsb::crc_ok(&f), "bit {bit}");
    }
    assert_eq!(adsb::df(&[]), 0);
    assert_eq!(adsb::df(&[0x28, 0, 0, 0, 0, 0, 0]), 5);
    assert_eq!(adsb::icao(&MODE_S[..3]), 0);
    assert_eq!(adsb::LONG_FRAME_BYTES, 14);
    assert_eq!(adsb::SHORT_FRAME_BYTES, 7);
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Header>();
    assert_send_sync::<Address>();
}
