//! The RDS character set of `sdrk_proto::rds` (IEC 62106, Annex E, Figure E.1): all 256 codes
//! against the table written out here, the round trip of `encode_char`, `RdsText` in the RDS
//! set, in UTF-8 and with the detection, and the raw text of `RdsState` next to its unchanged
//! ASCII view.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use sdrk_proto::Error;
use sdrk_proto::rds::{
    self, Charset, Group, RdsChars, RdsState, RdsText, decode_char, encode_char, map_char,
};

/// Figure E.1, one string of 16 characters per row (row = high nibble). Control codes and the
/// unused 0xFF are written as spaces.
const ROWS: [&str; 16] = [
    "                ",
    "                ",
    " !\"#¤%&'()*+,-./",
    "0123456789:;<=>?",
    "@ABCDEFGHIJKLMNO",
    "PQRSTUVWXYZ[\\]―_",
    "║abcdefghijklmno",
    "pqrstuvwxyz{|}¯ ",
    "áàéèíìóòúùÑÇŞß¡Ĳ",
    "âäêëîïôöûüñçşğıĳ",
    "ªα©‰Ğěňőπ€£$←↑→↓",
    "º¹²³±İńűµ¿÷°¼½¾§",
    "ÁÀÉÈÍÌÓÒÚÙŘČŠŽÐĿ",
    "ÂÄÊËÎÏÔÖÛÜřčšžđŀ",
    "ÃÅÆŒŷÝÕØÞŊŔĆŚŹŦð",
    "ãåæœŵýõøþŋŕćśźŧ ",
];

fn table() -> Vec<char> {
    let t: Vec<char> = ROWS.iter().flat_map(|r| r.chars()).collect();
    assert_eq!(t.len(), 256);
    t
}

/// Whether `b` is a control code or 0xFF (no character of its own).
fn is_control(b: u8) -> bool {
    b < 0x20 || b == 0x7F || b == 0xFF
}

#[test]
fn all_256_codes_decode_as_figure_e1() {
    // decode_char is const.
    const U: char = decode_char(0x99);
    for (b, want) in (0..=255u8).zip(table()) {
        assert_eq!(decode_char(b), want, "{b:#04x}");
    }
    // The places where the table is not ASCII.
    assert_eq!(decode_char(0x24), '\u{A4}');
    assert_eq!(decode_char(0x5E), '\u{2015}');
    assert_eq!(decode_char(0x60), '\u{2551}');
    assert_eq!(decode_char(0x7E), '\u{AF}');
    assert_eq!(decode_char(0xAB), '$');
    assert_eq!(decode_char(0xB8), '\u{B5}');
    assert_eq!(U, 'ü');
}

#[test]
fn every_character_round_trips_and_the_rest_has_no_code() {
    for b in 0..=255u8 {
        let c = decode_char(b);
        if is_control(b) {
            assert_eq!(c, ' ', "{b:#04x}");
            assert_eq!(encode_char(c), Some(0x20));
        } else {
            assert_eq!(encode_char(c), Some(b), "{b:#04x} {c}");
        }
    }
    // 223 distinct characters: 0x20 to 0x7E and 0x80 to 0xFE.
    let mut chars: Vec<char> = (0..=255u8)
        .filter(|&b| !is_control(b))
        .map(decode_char)
        .collect();
    chars.sort_unstable();
    chars.dedup();
    assert_eq!(chars.len(), 95 + 127);
    for c in ['^', '`', '~', '\n', '\r', '\u{FFFD}', 'Ä', '日'] {
        let want = (0..=255u8).find(|&b| !is_control(b) && decode_char(b) == c);
        assert_eq!(encode_char(c), want, "{c:?}");
    }
    assert_eq!(encode_char('^'), None);
    assert_eq!(encode_char('`'), None);
    assert_eq!(encode_char('~'), None);
    assert_eq!(encode_char('$'), Some(0xAB));
    assert_eq!(encode_char('\u{A4}'), Some(0x24));
}

#[test]
fn umlauts_of_real_stations() {
    let mut buf = [0u8; 64];
    let swr = b"SWR1 Baden-W\x99rttemberg";
    assert_eq!(
        RdsText::new(swr).to_utf8(&mut buf).unwrap(),
        "SWR1 Baden-Württemberg"
    );
    assert_eq!(
        RdsText::new(swr).to_ascii(&mut buf).unwrap(),
        "SWR1 Baden-W?rttemberg"
    );
    let t = RdsText::new(b"gr\x97\x8Dten");
    assert_eq!(t.to_utf8(&mut buf).unwrap(), "größten");
    assert_eq!(t.utf8_len(), "größten".len());
    assert_eq!(t.char_count(), 7);
    assert_eq!(t.to_string(), "größten");
    assert_eq!(t.chars().collect::<String>(), "größten");
    // encode_text is the inverse.
    let n = rds::encode_text("SWR1 Baden-Württemberg", &mut buf).unwrap();
    assert_eq!(&buf[..n], swr);
    let n = rds::encode_text("Preis: 5 $ / 4 €", &mut buf).unwrap();
    assert_eq!(&buf[..n], b"Preis: 5 \xAB / 4 \xA9");
}

#[test]
fn utf8_detection() {
    let mut buf = [0u8; 64];
    // Valid UTF-8 with a multi-byte sequence: UTF-8.
    let t = RdsText::new("Württemberg".as_bytes()).with_charset(Charset::Auto);
    assert_eq!(t.charset(), Charset::Utf8);
    assert_eq!(t.to_utf8(&mut buf).unwrap(), "Württemberg");
    assert_eq!(t.to_ascii(&mut buf).unwrap(), "W?rttemberg");
    assert_eq!(t.char_count(), 11);
    assert_eq!(t.to_string(), "Württemberg");
    // A lone 0xC3 (È in the table) is no valid UTF-8: the table.
    let t = RdsText::new(b"\xC3cole").with_charset(Charset::Auto);
    assert_eq!(t.charset(), Charset::Rds);
    assert_eq!(t.to_string(), "Ècole");
    // ASCII only: the table (0x24 is the currency sign there).
    let t = RdsText::new(b"5$").with_charset(Charset::Auto);
    assert_eq!(t.charset(), Charset::Rds);
    assert_eq!(t.to_string(), "5\u{A4}");
    // Forced UTF-8: invalid sequences become U+FFFD, control codes spaces.
    let t = RdsText::new(b"a\xC3b\xFF\n").with_charset(Charset::Utf8);
    assert_eq!(t.to_string(), "a\u{FFFD}b\u{FFFD} ");
    assert_eq!(t.char_count(), 5);
    assert_eq!(t.utf8_len(), 9);
    assert_eq!(t.to_ascii(&mut buf).unwrap(), "a?b? ");
    // Back to the table.
    assert_eq!(t.with_charset(Charset::Rds).to_string(), "aÈb  ");
    assert_eq!(
        RdsText::new(b"").with_charset(Charset::Auto).charset(),
        Charset::Rds
    );
    assert_eq!(Charset::default(), Charset::Rds);
}

#[test]
fn buffers_that_are_too_small_fail_without_truncating() {
    let t = RdsText::new(b"\x99ber");
    let mut buf = [0u8; 8];
    assert_eq!(
        t.to_utf8(&mut buf[..4]),
        Err(Error::BufferTooSmall { needed: 5, got: 4 })
    );
    assert_eq!(t.to_utf8(&mut buf[..5]).unwrap(), "über");
    assert_eq!(
        t.to_ascii(&mut buf[..3]),
        Err(Error::BufferTooSmall { needed: 4, got: 3 })
    );
    assert_eq!(t.to_ascii(&mut buf[..4]).unwrap(), "?ber");
    assert_eq!(RdsText::new(b"").to_utf8(&mut []).unwrap(), "");
    let mut out = [0xEEu8; 3];
    assert_eq!(
        rds::encode_text("äbcd", &mut out),
        Err(Error::BufferTooSmall { needed: 4, got: 3 })
    );
    assert_eq!(out, [0xEE; 3], "nothing written");
    assert_eq!(
        rds::encode_text("a~", &mut out),
        Err(Error::InvalidArgument(
            "character not in the rds character set"
        ))
    );
    assert_eq!(rds::encode_text("", &mut []), Ok(0));
}

fn ps_group(seg: u16, pair: [u8; 2]) -> Group {
    Group::new([0xD123, 0x0148 | seg, 0xE0E0, u16::from_be_bytes(pair)])
}

fn rt_group(seg: u16, text: [u8; 4]) -> Group {
    Group::new([
        0xD123,
        0x2140 | seg,
        u16::from_be_bytes([text[0], text[1]]),
        u16::from_be_bytes([text[2], text[3]]),
    ])
}

#[test]
fn rds_state_keeps_the_ascii_view_and_the_raw_bytes() {
    // An ASCII station: ps()/rt() as before, the texts equal them.
    let mut s = RdsState::new();
    for (seg, pair) in [(0, *b"SD"), (1, *b"RK"), (2, *b" T"), (3, *b"ST")] {
        assert!(s.ps_text().is_none());
        s.update(&ps_group(seg, pair));
    }
    assert_eq!(s.ps(), "SDRK TST");
    assert_eq!(s.ps_text().unwrap().to_string(), "SDRK TST");
    assert_eq!(s.ps_text().unwrap().bytes(), b"SDRK TST");
    assert!(s.rt_text().is_none());
    s.update(&rt_group(0, *b"Hi! "));
    s.update(&rt_group(1, *b"ok\r "));
    assert_eq!(s.rt(), "Hi! ok");
    assert_eq!(s.rt_text().unwrap().bytes(), b"Hi! ok");

    // Another station with umlauts and control codes.
    for (seg, pair) in [
        (0, *b"SW"),
        (1, *b"R1"),
        (2, [b' ', 0x99]),
        (3, [0x7F, 0x0D]),
    ] {
        s.update(&Group::new([
            0xD3C1,
            0x0148 | seg,
            0xE0E0,
            u16::from_be_bytes(pair),
        ]));
    }
    assert_eq!(s.ps(), "SWR1 ???");
    let ps = s.ps_text().unwrap();
    assert_eq!(ps.bytes(), b"SWR1 \x99\x7F\x0D");
    assert_eq!(ps.to_string(), "SWR1 ü  ");
    let text = *b"Die gr\x97\x8Dten Hits, 5 \xAB\r";
    for (seg, chunk) in text.chunks(4).enumerate() {
        let mut four = [b' '; 4];
        four[..chunk.len()].copy_from_slice(chunk);
        let mut g = rt_group(seg as u16, four);
        g.blocks[0] = 0xD3C1;
        s.update(&g);
    }
    assert_eq!(s.rt(), "Die gr??ten Hits, 5 ?");
    let rt = s.rt_text().unwrap();
    assert_eq!(rt.bytes(), &text[..text.len() - 1]);
    let mut buf = [0u8; 2 * rds::RT_LEN];
    assert_eq!(rt.to_utf8(&mut buf).unwrap(), "Die größten Hits, 5 $");
    assert_eq!(rt.to_ascii(&mut buf).unwrap(), "Die gr??ten Hits, 5 $");
    // ps()/rt() are map_char of the raw bytes.
    let mapped: Vec<u8> = s
        .rt_text()
        .unwrap()
        .bytes()
        .iter()
        .map(|&b| map_char(b))
        .collect();
    assert_eq!(s.rt().as_bytes(), mapped);

    // A UTF-8 station (against the standard): Auto finds it.
    let mut s = RdsState::new();
    let utf8 = "Grüße\r".as_bytes();
    let mut padded = [b' '; 8];
    padded[..utf8.len()].copy_from_slice(utf8);
    for (seg, chunk) in padded.chunks(4).enumerate() {
        s.update(&rt_group(seg as u16, chunk.try_into().unwrap()));
    }
    assert_eq!(s.rt(), "Gr????e");
    let rt = s.rt_text().unwrap();
    assert_eq!(rt.with_charset(Charset::Auto).to_string(), "Grüße");
    // 0xC3 0xBC 0xC3 0x9F in the RDS table: È ¼ È ĳ.
    assert_eq!(rt.to_string(), "GrÈ¼Èĳe");

    // reset clears the raw text as well.
    s.reset();
    assert!(s.rt_text().is_none());
    assert_eq!(s, RdsState::new());
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn new_public_types_are_send_and_sync() {
    assert_send_sync::<Charset>();
    assert_send_sync::<RdsText<'static>>();
    assert_send_sync::<RdsChars<'static>>();
}
