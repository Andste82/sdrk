//! JSON Lines packet files (M37 §5): round trip, the exact line format and the error lines.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_possible_truncation
)]

use std::io::{BufReader, Cursor};

use sdrk_device::{ErrorKind, SdrkError};
use sdrk_io::Error;
use sdrk_io::packets::{MAX_DATA_BYTES, PacketRecord, from_hex, read_jsonl, to_hex, write_jsonl};
use sdrk_testkit::Pcg32;

fn written(packets: &[PacketRecord]) -> String {
    let mut out = Vec::new();
    write_jsonl(&mut out, packets).unwrap();
    String::from_utf8(out).unwrap()
}

fn error_of(text: &str) -> (String, String) {
    match read_jsonl(text.as_bytes()).unwrap_err() {
        Error::Parse { file, reason } => (file, reason),
        other => panic!("not a parse error: {other:?}"),
    }
}

#[test]
fn the_line_format_is_exact() {
    let packets = [
        PacketRecord::new(123, 1, vec![0x1a, 0xcf]).unwrap(),
        PacketRecord::new(0, 0, vec![]).unwrap(),
        PacketRecord::new(u64::MAX, u32::MAX, vec![0x00, 0x0f, 0xf0, 0xff, 0x7e]).unwrap(),
    ];
    assert_eq!(
        written(&packets),
        "{\"index\":123,\"flags\":1,\"data\":\"1acf\"}\n\
         {\"index\":0,\"flags\":0,\"data\":\"\"}\n\
         {\"index\":18446744073709551615,\"flags\":4294967295,\"data\":\"000ff0ff7e\"}\n"
    );
    assert_eq!(written(&[]), "");
}

#[test]
fn round_trip_of_random_packets() {
    let mut rng = Pcg32::new(479, 0);
    let mut packets = Vec::new();
    for i in 0..500u64 {
        let len = if i == 0 {
            MAX_DATA_BYTES
        } else {
            rng.below(MAX_DATA_BYTES as u32 + 1) as usize
        };
        let data: Vec<u8> = (0..len).map(|_| rng.below(256) as u8).collect();
        let index = rng.next_u64();
        let flags = rng.next_u32();
        packets.push(PacketRecord::new(index, flags, data).unwrap());
    }
    let text = written(&packets);
    assert_eq!(text.lines().count(), packets.len());
    let back = read_jsonl(BufReader::new(Cursor::new(text.as_bytes()))).unwrap();
    assert_eq!(back, packets);
    // Every line carries lower-case hex only.
    assert!(!text.chars().any(|c| c.is_ascii_uppercase()));
}

#[test]
fn reader_accepts_empty_lines_crlf_and_any_field_order() {
    let text = "\n{\"data\":\"ff\",\"flags\":2,\"index\":5}\r\n   \n{\"index\":6,\"flags\":0,\"data\":\"\"}";
    let p = read_jsonl(text.as_bytes()).unwrap();
    assert_eq!(p.len(), 2);
    assert_eq!(
        (p[0].index(), p[0].flags(), p[0].data()),
        (5, 2, &[0xff][..])
    );
    assert_eq!((p[1].index(), p[1].data()), (6, &[][..]));
    assert_eq!(
        read_jsonl("".as_bytes()).unwrap(),
        Vec::<PacketRecord>::new()
    );
}

#[test]
fn error_lines_name_the_line_number() {
    let good = "{\"index\":1,\"flags\":0,\"data\":\"00\"}";
    let cases: &[(&str, &str)] = &[
        (
            r#"{"index":1,"flags":0,"data":"00","extra":true}"#,
            "unknown field `extra`",
        ),
        (r#"{"index":1,"flags":0}"#, "missing field `data`"),
        (r#"{"flags":0,"data":""}"#, "missing field `index`"),
        (r#"{"index":1,"data":""}"#, "missing field `flags`"),
        (
            r#"{"index":-1,"flags":0,"data":""}"#,
            "`index` must be an unsigned 64-bit integer",
        ),
        (
            r#"{"index":1.5,"flags":0,"data":""}"#,
            "`index` must be an unsigned 64-bit integer",
        ),
        (
            r#"{"index":"1","flags":0,"data":""}"#,
            "`index` must be an unsigned 64-bit integer",
        ),
        (
            r#"{"index":1,"flags":4294967296,"data":""}"#,
            "`flags` must be an unsigned 32-bit integer",
        ),
        (
            r#"{"index":1,"flags":0,"data":7}"#,
            "`data` must be a string",
        ),
        (
            r#"{"index":1,"flags":0,"data":"1ACF"}"#,
            "data must be lower-case hex digits (0-9, a-f) without spaces",
        ),
        (
            r#"{"index":1,"flags":0,"data":"1a cf"}"#,
            "data has an odd number of hex digits (5)",
        ),
        (
            r#"{"index":1,"flags":0,"data":"1a  cf"}"#,
            "data must be lower-case hex digits (0-9, a-f) without spaces",
        ),
        (
            r#"{"index":1,"flags":0,"data":"1g"}"#,
            "data must be lower-case hex digits (0-9, a-f) without spaces",
        ),
        (
            r#"{"index":1,"flags":0,"data":"abc"}"#,
            "data has an odd number of hex digits (3)",
        ),
        ("[1,2,3]", "a line must be a JSON object"),
    ];
    for (bad, reason) in cases {
        // The bad line is line 3: line 1 is good, line 2 is empty.
        let text = format!("{good}\n\n{bad}\n{good}\n");
        let (file, got) = error_of(&text);
        assert_eq!(file, "line 3", "{bad}");
        assert_eq!(got, *reason, "{bad}");
    }
    let (file, reason) = error_of("{\"index\":1,");
    assert_eq!(file, "line 1");
    assert!(reason.starts_with("invalid JSON"), "{reason}");
}

#[test]
fn data_longer_than_256_bytes_is_rejected_with_the_line() {
    let max = format!(
        "{{\"index\":0,\"flags\":0,\"data\":\"{}\"}}\n",
        "ab".repeat(MAX_DATA_BYTES)
    );
    assert_eq!(
        read_jsonl(max.as_bytes()).unwrap()[0].data().len(),
        MAX_DATA_BYTES
    );
    let long = format!(
        "{max}{{\"index\":0,\"flags\":0,\"data\":\"{}\"}}\n",
        "ab".repeat(MAX_DATA_BYTES + 1)
    );
    let err = read_jsonl(long.as_bytes()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Parse);
    assert_eq!(
        err.to_string(),
        "cannot parse line 2: data has 257 bytes, at most 256 are allowed"
    );
}

#[test]
fn non_utf8_input_is_an_io_error() {
    let err = read_jsonl(&[0xff, 0xfe, b'\n'][..]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Io);
}

#[test]
fn hex_helpers() {
    let all: Vec<u8> = (0..=255u8).collect();
    let hex = to_hex(&all);
    assert_eq!(hex.len(), 512);
    assert!(hex.starts_with("000102") && hex.ends_with("fdfeff"));
    assert_eq!(from_hex(&hex).unwrap(), all);
    assert_eq!(from_hex("").unwrap(), Vec::<u8>::new());
    let e = from_hex("zz").unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidArgument);
    let e = PacketRecord::new(0, 0, vec![0; MAX_DATA_BYTES + 1]).unwrap_err();
    assert_eq!(
        e.to_string(),
        "invalid argument: data has 257 bytes, at most 256 are allowed"
    );
    assert_eq!(e.kind(), ErrorKind::InvalidArgument);
}

#[test]
fn writer_reports_io_errors() {
    struct Broken;
    impl std::io::Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("disk full"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let p = [PacketRecord::new(0, 0, b"x".to_vec()).unwrap()];
    let err = write_jsonl(Broken, &p).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Io);
    assert!(err.to_string().contains("disk full"));
}
