//! Round trips and behaviour of the raw, SigMF and WAV streams, and the conformance checks.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::doc_markdown,
    clippy::unreadable_literal,
    clippy::needless_pass_by_value,
    clippy::redundant_closure_for_method_calls
)]

use std::io::Cursor;

use sdrk_device::{
    BlockFlags, BlockMeta, ErrorKind, Rate, RxStream, SampleFormat, SampleIndex, SdrkError,
    TimeAnchor, TimeSource, TxStream, formats,
};
use sdrk_io::{
    Annotation, Error, RawReader, RawWriter, SigmfReader, SigmfWriter, WavReader, WavWriter,
};
use sdrk_testkit::conformance;

fn rate(hz: u64) -> Rate {
    Rate::hz(hz).unwrap()
}

/// `n` bytes of a repeating pattern.
fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 7 % 253) as u8).collect()
}

fn read_all(r: &mut impl RxStream<Error = Error>, buf_len: usize) -> (Vec<u8>, Vec<BlockMeta>) {
    let mut bytes = Vec::new();
    let mut metas = Vec::new();
    let mut buf = vec![0u8; buf_len];
    loop {
        let (n, meta) = r.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..n]);
        metas.push(meta);
    }
    (bytes, metas)
}

// ------------------------------------------------------------------------------------- errors

#[test]
fn errors_have_kinds_and_lowercase_messages() {
    let io: Error = std::io::Error::new(std::io::ErrorKind::NotFound, "gone").into();
    assert_eq!(io.kind(), ErrorKind::Io);
    let parse = Error::Parse {
        file: "a.sigmf-meta".into(),
        reason: "bad".into(),
    };
    assert_eq!(parse.kind(), ErrorKind::Parse);
    assert_eq!(Error::Unsupported("x").kind(), ErrorKind::Unsupported);
    assert_eq!(
        Error::InvalidArgument("x".into()).kind(),
        ErrorKind::InvalidArgument
    );
    let dev: Error = sdrk_device::Error::Overflow("x").into();
    assert_eq!(dev.kind(), ErrorKind::OutOfRange);
    assert!(std::error::Error::source(&dev).is_some());
    assert!(std::error::Error::source(&io).is_none());
    for e in [&io, &parse, &dev, &Error::Unsupported("feature")] {
        let text = e.to_string();
        assert!(text.chars().next().unwrap().is_lowercase(), "{text}");
        assert!(!text.ends_with('.'));
    }
    assert_eq!(parse.clone(), parse);
}

#[test]
fn public_types_are_send() {
    fn send<T: Send>() {}
    send::<Error>();
    send::<RawReader<std::fs::File>>();
    send::<RawWriter<Vec<u8>>>();
    send::<SigmfReader<std::fs::File>>();
    send::<SigmfWriter<Vec<u8>>>();
    send::<WavReader<std::fs::File>>();
    send::<WavWriter<Cursor<Vec<u8>>>>();
    send::<Annotation>();
    send::<sdrk_io::Capture>();
    send::<sdrk_io::FlagCounts>();
}

// ---------------------------------------------------------------------------------------- raw

const RAW_FORMATS: [SampleFormat; 12] = [
    formats::CU8,
    formats::CS8,
    formats::CS16,
    formats::CS12_LSB16,
    formats::SC16_Q11,
    formats::CS16_PLANAR,
    formats::PACKED12,
    formats::PACKED10,
    formats::I2S32_MSB,
    formats::CF32,
    formats::RF32,
    formats::RS16,
];

#[test]
fn raw_round_trip_for_every_format() {
    for format in RAW_FORMATS {
        // 8 samples fill the packing groups of every format
        let sample_bytes = format.bytes_for_samples(8).unwrap();
        let data = pattern(sample_bytes * 13);
        let mut w = RawWriter::new(Vec::new(), format, rate(1000));
        assert_eq!(w.format(), format);
        assert_eq!(w.sample_rate(), rate(1000));
        for piece in data.chunks(sample_bytes) {
            w.write(piece, &BlockMeta::default()).unwrap();
        }
        assert_eq!(w.samples_written(), 8 * 13);
        let bytes = w.into_inner();
        assert_eq!(bytes, data);
        let mut r = RawReader::new(&bytes[..], format, rate(1000), SampleIndex(0), 24).unwrap();
        let (back, metas) = read_all(&mut r, 100_000);
        assert_eq!(back, data, "{format:?}");
        // chunks of 24 samples; indices continue
        let mut expect = 0u64;
        for m in &metas {
            assert_eq!(m.start.0, expect);
            expect += 24;
        }
    }
}

#[test]
fn raw_reader_semantics() {
    let data = pattern(21); // 10 CS8 samples and one stray byte
    let anchor = TimeAnchor {
        index: SampleIndex(1000),
        unix_ns: 5,
        source: TimeSource::Host,
    };
    let mut r = RawReader::new(&data[..], formats::CS8, rate(48_000), SampleIndex(1000), 4)
        .unwrap()
        .with_anchor(anchor);
    let mut buf = [0u8; 64];
    let (n, m) = r.read(&mut buf).unwrap();
    assert_eq!((n, m.start.0, m.anchor), (8, 1000, Some(anchor)));
    let (n, m) = r.read(&mut buf).unwrap();
    assert_eq!((n, m.start.0, m.anchor), (8, 1004, None));
    let (n, m) = r.read(&mut buf).unwrap();
    assert_eq!((n, m.start.0), (4, 1008)); // two samples; the stray byte is dropped
    assert_eq!(r.next_index(), SampleIndex(1010));
    assert_eq!(r.read(&mut buf).unwrap().0, 0);
    assert_eq!(r.read(&mut buf).unwrap().0, 0);
    // a small buffer limits the block; below one sample it is an error
    let mut r = RawReader::new(&data[..], formats::CS16, rate(1), SampleIndex(0), 100).unwrap();
    assert_eq!(r.read(&mut buf[..7]).unwrap().0, 4);
    assert_eq!(
        r.read(&mut buf[..3]).unwrap_err().kind(),
        ErrorKind::BufferTooSmall
    );
    // invalid chunk size: smaller than a packing group
    assert!(RawReader::new(&data[..], formats::PACKED10, rate(1), SampleIndex(0), 1).is_err());
    assert!(RawReader::new(&data[..], formats::CS8, rate(1), SampleIndex(0), 0).is_err());
    // the file reader returns an anchor only when there is data
    let mut empty = RawReader::new(&[][..], formats::CS8, rate(1), SampleIndex(0), 4)
        .unwrap()
        .with_anchor(anchor);
    assert_eq!(empty.read(&mut buf).unwrap().1.anchor, None);
    let _ = empty.into_inner();
}

/// A reader that returns one byte at a time and one `Interrupted`.
struct Dribble<'a> {
    data: &'a [u8],
    interrupted: bool,
}

impl std::io::Read for Dribble<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if !self.interrupted {
            self.interrupted = true;
            return Err(std::io::ErrorKind::Interrupted.into());
        }
        let Some((&first, rest)) = self.data.split_first() else {
            return Ok(0);
        };
        buf[0] = first;
        self.data = rest;
        Ok(1)
    }
}

#[test]
fn raw_reader_assembles_whole_samples_from_short_reads() {
    let data = pattern(40);
    let mut r = RawReader::new(
        Dribble {
            data: &data,
            interrupted: false,
        },
        formats::CF32,
        rate(10),
        SampleIndex(0),
        3,
    )
    .unwrap();
    let (back, metas) = read_all(&mut r, 1000);
    assert_eq!(back, data[..40]);
    assert_eq!(metas.len(), 2); // 12 + 12 + ... 40 bytes = 5 samples: chunks of 3 and 2
}

#[test]
fn raw_writer_counts_flags_and_rejects_partial_samples() {
    let mut w = RawWriter::new(Vec::new(), formats::CS16, rate(1));
    let flags = [
        BlockFlags::GAP_BEFORE,
        BlockFlags::OVERFLOW,
        BlockFlags::UNDERRUN,
        BlockFlags::BURST_START,
        BlockFlags::BURST_END,
        BlockFlags::LATE,
        BlockFlags::RETUNED,
        BlockFlags::LATE.union(BlockFlags::BURST_END),
    ];
    for f in flags {
        let meta = BlockMeta {
            flags: f,
            ..BlockMeta::default()
        };
        w.write(&[0, 0, 0, 0], &meta).unwrap();
    }
    let c = w.flag_counts();
    assert_eq!(
        (
            c.gap_before,
            c.overflow,
            c.underrun,
            c.burst_start,
            c.burst_end,
            c.late,
            c.retuned
        ),
        (1, 1, 1, 1, 2, 2, 1)
    );
    assert!(w.write(&[0, 0, 0], &BlockMeta::default()).is_err());
    assert_eq!(w.poll_status().unwrap(), None);
    // the timing of a burst is ignored
    let timed = BlockMeta {
        tx_time_ns: Some(1),
        ..BlockMeta::default()
    };
    assert_eq!(w.write(&[0, 0, 0, 0], &timed).unwrap(), 4);
}

#[test]
fn raw_writer_reports_io_errors() {
    struct Broken;
    impl std::io::Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("disk full"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut w = RawWriter::new(Broken, formats::CS8, rate(1));
    let e = w.write(&[0, 0], &BlockMeta::default()).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Io);
    assert!(e.to_string().contains("disk full"));
}

// -------------------------------------------------------------------------------------- SigMF

const SIGMF_FORMATS: [SampleFormat; 6] = [
    formats::CU8,
    formats::CS8,
    formats::CS16,
    formats::CF32,
    formats::RS16,
    formats::RF32,
];

#[test]
fn sigmf_round_trip_for_every_format_through_files() {
    let dir = tempfile::tempdir().unwrap();
    for (i, format) in SIGMF_FORMATS.into_iter().enumerate() {
        let sb = format.bytes_for_samples(1).unwrap();
        let base = dir.path().join(format!("rec{i}"));
        let mut w = SigmfWriter::create(&base, format, rate(2_400_000)).unwrap();
        w.set_frequency(433.92e6).unwrap();
        let data = pattern(sb * 1000);
        let anchor = TimeAnchor {
            index: SampleIndex(0),
            unix_ns: 1_700_000_000_000_000_000,
            source: TimeSource::Host,
        };
        w.write(
            &data[..sb * 400],
            &BlockMeta {
                anchor: Some(anchor),
                ..BlockMeta::default()
            },
        )
        .unwrap();
        w.write(
            &data[sb * 400..],
            &BlockMeta {
                start: SampleIndex(400),
                ..BlockMeta::default()
            },
        )
        .unwrap();
        w.annotate(Annotation {
            sample_start: 10,
            sample_count: 20,
            label: Some("größe \"quoted\"".into()),
            ..Annotation::default()
        })
        .unwrap();
        let (_file, text) = w.finish().unwrap();
        assert!(text.ends_with("}\n") && !text.contains('\r'));
        // the meta file on disk is the returned text
        let meta_path = dir.path().join(format!("rec{i}.sigmf-meta"));
        assert_eq!(std::fs::read_to_string(&meta_path).unwrap(), text);
        // open by base path, by meta path and by data path
        for variant in [
            base.clone(),
            meta_path.clone(),
            dir.path().join(format!("rec{i}.sigmf-data")),
        ] {
            let mut r = SigmfReader::open(&variant).unwrap();
            assert_eq!(r.format(), format);
            assert_eq!(r.sample_rate(), rate(2_400_000));
            assert_eq!(r.annotations().len(), 1);
            assert_eq!(
                r.annotations()[0].label.as_deref(),
                Some("größe \"quoted\"")
            );
            let (bytes, metas) = read_all(&mut r, 1 << 16);
            assert_eq!(bytes, data);
            assert_eq!(metas.len(), 1, "no new capture for contiguous blocks");
            assert_eq!(metas[0].anchor.unwrap().unix_ns, 1_700_000_000_000_000_000);
            assert_eq!(metas[0].flags.bits(), 0);
        }
    }
}

#[test]
fn sigmf_gaps_retunes_and_chunks() {
    let mut w = SigmfWriter::new(Vec::new(), formats::CS8, rate(1000)).unwrap();
    let blk = |start: u64, flags: BlockFlags| BlockMeta {
        start: SampleIndex(start),
        flags,
        ..BlockMeta::default()
    };
    w.set_frequency(1e8).unwrap();
    w.write(&[1; 20], &blk(0, BlockFlags::empty())).unwrap(); // 10 samples
    w.write(&[2; 20], &blk(10, BlockFlags::OVERFLOW)).unwrap(); // continues, flag not stored
    w.write(&[3; 20], &blk(50, BlockFlags::empty())).unwrap(); // gap of 30
    w.write(&[4; 20], &blk(60, BlockFlags::RETUNED)).unwrap(); // retune flag: new capture
    w.set_frequency(2e8).unwrap();
    w.write(&[5; 20], &blk(70, BlockFlags::empty())).unwrap(); // frequency changed
    w.write(&[6; 2], &blk(80, BlockFlags::GAP_BEFORE)).unwrap(); // flag without jump
    assert!(w.write(&[0; 2], &blk(79, BlockFlags::empty())).is_err()); // before the end
    assert_eq!(w.write(&[], &blk(500, BlockFlags::empty())).unwrap(), 0);
    assert_eq!(w.samples_written(), 51);
    assert_eq!(w.flag_counts().overflow, 1);
    let (data, text) = w.finish().unwrap();
    let mut r = SigmfReader::new(&text, &data[..])
        .unwrap()
        .with_chunk_samples(4)
        .unwrap();
    assert_eq!(r.captures().len(), 5);
    let mut buf = vec![0u8; 1000];
    let mut got = Vec::new();
    loop {
        let (n, m) = r.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        got.push((m.start.0, n / 2, m.flags));
    }
    let gap = BlockFlags::GAP_BEFORE;
    let retuned = BlockFlags::RETUNED;
    assert_eq!(
        got,
        vec![
            (0, 4, BlockFlags::empty()),
            (4, 4, BlockFlags::empty()),
            (8, 4, BlockFlags::empty()),
            (12, 4, BlockFlags::empty()),
            (16, 4, BlockFlags::empty()),
            (50, 4, gap),
            (54, 4, BlockFlags::empty()),
            (58, 2, BlockFlags::empty()),
            (60, 4, BlockFlags::empty()), // same frequency: no RETUNED
            (64, 4, BlockFlags::empty()),
            (68, 2, BlockFlags::empty()),
            (70, 4, retuned), // 1e8 -> 2e8
            (74, 4, BlockFlags::empty()),
            (78, 2, BlockFlags::empty()),
            (80, 1, BlockFlags::empty()), // the stray GAP_BEFORE did not create a jump
        ]
    );
}

fn meta_with(global: &str, captures: &str) -> String {
    format!(
        r#"{{"global": {{"core:datatype": "cu8", "core:sample_rate": 1000, "core:version": "1.0.0" {global} }},
            "captures": {captures}, "annotations": [] }}"#
    )
}

#[test]
fn sigmf_sample_rates_are_exact_or_rejected() {
    let ok = |text: &str| -> Rate {
        let m = format!(
            r#"{{"global": {{"core:datatype": "cu8", "core:sample_rate": {text}, "core:version": "1.0.0"}}, "captures": [], "annotations": []}}"#
        );
        SigmfReader::new(&m, &[][..]).unwrap().sample_rate()
    };
    assert_eq!(ok("2400000"), rate(2_400_000));
    assert_eq!(ok("2.4e6"), rate(2_400_000));
    assert_eq!(ok("2400000.0"), rate(2_400_000));
    assert_eq!(ok("44100.5"), Rate::new(88_201, 2).unwrap());
    assert_eq!(ok("0.5"), Rate::new(1, 2).unwrap());
    assert_eq!(ok("1e-6"), Rate::new(1, 1_000_000).unwrap());
    assert_eq!(
        ok("1000000.123456"),
        Rate::new(1_000_000_123_456, 1_000_000).unwrap()
    );
    for bad in ["0.1234567", "0", "-5", "1e300", "0.3333333333333333"] {
        let m = format!(
            r#"{{"global": {{"core:datatype": "cu8", "core:sample_rate": {bad}, "core:version": "1.0.0"}}}}"#
        );
        let e = SigmfReader::new(&m, &[][..]).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidArgument, "{bad}");
    }
    // a string is not a number
    let m = r#"{"global": {"core:datatype": "cu8", "core:sample_rate": "1000", "core:version": "1.0.0"}}"#;
    assert_eq!(
        SigmfReader::new(m, &[][..]).unwrap_err().kind(),
        ErrorKind::Parse
    );
    // the writer refuses a rate that SigMF cannot hold exactly
    assert!(SigmfWriter::new(Vec::new(), formats::CU8, Rate::new(1, 3).unwrap()).is_err());
    assert!(SigmfWriter::new(Vec::new(), formats::CU8, Rate::new(1, 1_000_000).unwrap()).is_ok());
}

#[test]
fn sigmf_datatypes() {
    for (name, format) in [
        ("cu8", formats::CU8),
        ("ci8", formats::CS8),
        ("ci16_le", formats::CS16),
        ("cf32_le", formats::CF32),
        ("ri16_le", formats::RS16),
        ("rf32_le", formats::RF32),
    ] {
        assert_eq!(sdrk_io::format_of_datatype(name).unwrap(), format);
        assert_eq!(sdrk_io::datatype_of_format(&format).unwrap(), name);
    }
    for name in [
        "cf32_be", "ci16_be", "ru8", "ri8", "ci32_le", "cf64_le", "", "CU8",
    ] {
        let e = sdrk_io::format_of_datatype(name).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::Unsupported, "{name}");
    }
    for format in [
        formats::CS12_LSB16,
        formats::PACKED12,
        formats::CS16_PLANAR,
        formats::I2S32_MSB,
    ] {
        assert!(sdrk_io::datatype_of_format(&format).is_err());
        assert!(SigmfWriter::new(Vec::new(), format, rate(1)).is_err());
    }
}

#[test]
fn sigmf_reader_rejects_malformed_and_unsupported_metadata() {
    let kind = |text: &str| SigmfReader::new(text, &[][..]).unwrap_err().kind();
    assert_eq!(kind("not json"), ErrorKind::Parse);
    assert_eq!(kind("[]"), ErrorKind::Parse);
    assert_eq!(kind("{}"), ErrorKind::Parse);
    assert_eq!(kind(r#"{"global": 3}"#), ErrorKind::Parse);
    assert_eq!(
        kind(r#"{"global": {"core:sample_rate": 1, "core:version": "1.0.0"}}"#),
        ErrorKind::Parse
    );
    assert_eq!(
        kind(r#"{"global": {"core:datatype": "cu8", "core:version": "1.0.0"}}"#),
        ErrorKind::Parse
    );
    assert_eq!(
        kind(r#"{"global": {"core:datatype": "cu8", "core:sample_rate": 1}}"#),
        ErrorKind::Parse
    );
    assert_eq!(
        kind(
            r#"{"global": {"core:datatype": "cu8", "core:sample_rate": 1, "core:version": "2.0.0"}}"#
        ),
        ErrorKind::Unsupported
    );
    assert_eq!(
        kind(&meta_with(r#", "core:dataset": "other.bin""#, "[]")),
        ErrorKind::Unsupported
    );
    assert_eq!(
        kind(&meta_with(r#", "core:metadata_only": true"#, "[]")),
        ErrorKind::Unsupported
    );
    assert_eq!(
        kind(&meta_with(
            "",
            r#"[{"core:sample_start": 0, "core:header_bytes": 4}]"#
        )),
        ErrorKind::Unsupported
    );
    assert_eq!(
        kind(&meta_with("", r#"[{"core:sample_start": 5}]"#)),
        ErrorKind::Parse
    );
    assert_eq!(
        kind(&meta_with(
            "",
            r#"[{"core:sample_start": 0}, {"core:sample_start": 9}, {"core:sample_start": 3}]"#
        )),
        ErrorKind::Parse
    );
    assert_eq!(
        kind(&meta_with("", r#"[{"core:sample_start": -1}]"#)),
        ErrorKind::Parse
    );
    assert_eq!(
        kind(&meta_with(
            "",
            r#"[{"core:sample_start": 0, "core:datetime": "yesterday"}]"#
        )),
        ErrorKind::Parse
    );
    assert_eq!(
        kind(&meta_with(
            "",
            r#"[{"core:sample_start": 0, "core:frequency": "fast"}]"#
        )),
        ErrorKind::Parse
    );
    assert_eq!(kind(&meta_with("", "{}")), ErrorKind::Parse);
    assert_eq!(kind(&meta_with("", "[3]")), ErrorKind::Parse);
    let bad_annotation = r#"{"global": {"core:datatype": "cu8", "core:sample_rate": 1, "core:version": "1.0.0"},
        "captures": [], "annotations": [{"core:sample_start": 0, "core:label": 7}]}"#;
    assert_eq!(kind(bad_annotation), ErrorKind::Parse);
    // a missing global_index equals sample_start, and a missing captures array is one capture
    let ok =
        r#"{"global": {"core:datatype": "cu8", "core:sample_rate": 1, "core:version": "1.2.0"}}"#;
    let r = SigmfReader::new(ok, &[][..]).unwrap();
    assert_eq!(r.captures().len(), 1);
    assert_eq!(r.captures()[0].global_index, 0);
}

#[test]
fn sigmf_global_index_going_backwards_fails_when_read() {
    let meta = meta_with(
        "",
        r#"[{"core:sample_start": 0, "core:global_index": 100}, {"core:sample_start": 4, "core:global_index": 50}]"#,
    );
    let data = [0u8; 16]; // 8 CU8 samples
    let mut r = SigmfReader::new(&meta, &data[..]).unwrap();
    let mut buf = [0u8; 64];
    let (n, m) = r.read(&mut buf).unwrap();
    assert_eq!((n, m.start.0), (8, 100));
    assert_eq!(r.read(&mut buf).unwrap_err().kind(), ErrorKind::Parse);
}

#[test]
fn sigmf_reader_survives_truncated_data_and_small_buffers() {
    let meta = meta_with(
        "",
        r#"[{"core:sample_start": 0}, {"core:sample_start": 100, "core:global_index": 500}]"#,
    );
    let data = pattern(45); // 22 samples and a stray byte; the second capture never starts
    let mut r = SigmfReader::new(&meta, &data[..]).unwrap();
    let mut buf = [0u8; 64];
    let (n, m) = r.read(&mut buf).unwrap();
    assert_eq!((n, m.start.0), (44, 0));
    assert_eq!(r.read(&mut buf).unwrap().0, 0);
    let mut r = SigmfReader::new(&meta, &data[..]).unwrap();
    assert_eq!(
        r.read(&mut buf[..1]).unwrap_err().kind(),
        ErrorKind::BufferTooSmall
    );
    assert!(
        SigmfReader::new(&meta, &data[..])
            .unwrap()
            .with_chunk_samples(0)
            .is_err()
    );
}

#[test]
fn sigmf_files_report_missing_files_and_bad_encoding() {
    let dir = tempfile::tempdir().unwrap();
    let e = SigmfReader::open(dir.path().join("absent")).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Io);
    std::fs::write(dir.path().join("x.sigmf-meta"), "{ broken").unwrap();
    std::fs::write(dir.path().join("x.sigmf-data"), []).unwrap();
    let e = SigmfReader::open(dir.path().join("x")).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Parse);
    assert!(e.to_string().contains("x.sigmf-meta"));
    // data file missing
    std::fs::write(
        dir.path().join("y.sigmf-meta"),
        r#"{"global": {"core:datatype": "cu8", "core:sample_rate": 1, "core:version": "1.0.0"}}"#,
    )
    .unwrap();
    assert_eq!(
        SigmfReader::open(dir.path().join("y")).unwrap_err().kind(),
        ErrorKind::Io
    );
    assert_eq!(
        SigmfWriter::create(
            dir.path().join("no_such_dir").join("z"),
            formats::CU8,
            rate(1)
        )
        .unwrap_err()
        .kind(),
        ErrorKind::Io
    );
}

#[test]
fn sigmf_writer_validates_inputs() {
    let mut w = SigmfWriter::new(Vec::new(), formats::CF32, rate(10)).unwrap();
    assert!(w.set_frequency(f64::NAN).is_err());
    assert!(
        w.annotate(Annotation {
            freq_lower_edge: Some(f64::INFINITY),
            ..Annotation::default()
        })
        .is_err()
    );
    assert!(w.write(&[0; 7], &BlockMeta::default()).is_err());
    assert_eq!(w.format(), formats::CF32);
    assert_eq!(w.sample_rate(), rate(10));
    assert_eq!(w.poll_status().unwrap(), None);
    // nothing written: one valid, empty document
    let (_, text) = w.finish().unwrap();
    assert!(text.contains("\"captures\": []"));
    assert!(text.contains("\"annotations\": []"));
}

#[test]
fn sigmf_anchor_time_is_extrapolated_to_the_capture_start() {
    // The anchor names sample 1000 but the block starts at 1010: datetime is that of 1010.
    let mut w = SigmfWriter::new(Vec::new(), formats::CU8, rate(1_000_000)).unwrap();
    let anchor = TimeAnchor {
        index: SampleIndex(1000),
        unix_ns: 5_000_000,
        source: TimeSource::Device,
    };
    let meta = BlockMeta {
        start: SampleIndex(1010),
        anchor: Some(anchor),
        ..BlockMeta::default()
    };
    w.write(&[0; 4], &meta).unwrap();
    let (data, text) = w.finish().unwrap();
    let mut r = SigmfReader::new(&text, &data[..]).unwrap();
    let mut buf = [0u8; 8];
    let (_, m) = r.read(&mut buf).unwrap();
    let a = m.anchor.unwrap();
    assert_eq!((a.index.0, a.unix_ns), (1010, 5_010_000));
    assert_eq!(a.source, TimeSource::Host);
}

// ---------------------------------------------------------------------------------------- WAV

#[test]
fn wav_round_trip_for_every_format() {
    for (format, hz) in [
        (formats::RS16, 8000),
        (formats::CS16, 48_000),
        (formats::RF32, 44_100),
        (formats::CF32, 96_000),
    ] {
        let iq = format.values_per_sample() == 2;
        let sb = format.bytes_for_samples(1).unwrap();
        let data = pattern(sb * 100);
        let mut w = WavWriter::new(Cursor::new(Vec::new()), format, rate(hz)).unwrap();
        w.write(&data[..sb * 30], &BlockMeta::default()).unwrap();
        w.write(&data[sb * 30..], &BlockMeta::default()).unwrap();
        let file = w.finish().unwrap().into_inner();
        assert_eq!(file.len(), 44 + data.len());
        let size = u32::from_le_bytes(file[4..8].try_into().unwrap());
        assert_eq!(size as usize, file.len() - 8);
        let mut r = WavReader::new(&file[..], iq, 17).unwrap();
        assert_eq!((r.format(), r.sample_rate()), (format, rate(hz)));
        let (back, metas) = read_all(&mut r, 1 << 16);
        assert_eq!(back, data);
        assert!(metas.iter().all(|m| m.anchor.is_none()));
        assert_eq!(metas[1].start.0, 17);
    }
}

#[test]
fn wav_reader_rejects_what_it_cannot_read() {
    fn header(tag: u16, channels: u16, bits: u16, rate: u32) -> Vec<u8> {
        let align = channels * bits / 8;
        let mut h = Vec::new();
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&40u32.to_le_bytes());
        h.extend_from_slice(b"WAVEfmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        for v in [tag, channels] {
            h.extend_from_slice(&v.to_le_bytes());
        }
        h.extend_from_slice(&rate.to_le_bytes());
        h.extend_from_slice(&(rate * u32::from(align)).to_le_bytes());
        h.extend_from_slice(&align.to_le_bytes());
        h.extend_from_slice(&bits.to_le_bytes());
        h.extend_from_slice(b"data");
        h.extend_from_slice(&4u32.to_le_bytes());
        h.extend_from_slice(&[0; 4]);
        h
    }
    let kind = |bytes: &[u8], iq: bool| WavReader::new(bytes, iq, 8).unwrap_err().kind();
    assert_eq!(
        kind(&header(1, 1, 16, 8000), true),
        ErrorKind::InvalidArgument
    ); // mono + iq
    assert_eq!(
        kind(&header(1, 2, 16, 8000), false),
        ErrorKind::InvalidArgument
    ); // stereo without iq
    assert_eq!(kind(&header(1, 1, 8, 8000), false), ErrorKind::Unsupported); // 8-bit
    assert_eq!(kind(&header(1, 1, 24, 8000), false), ErrorKind::Unsupported);
    assert_eq!(kind(&header(1, 1, 32, 8000), false), ErrorKind::Unsupported); // 32-bit integer
    assert_eq!(kind(&header(3, 1, 64, 8000), false), ErrorKind::Unsupported);
    assert_eq!(kind(&header(1, 3, 16, 8000), false), ErrorKind::Unsupported);
    assert_eq!(kind(&header(7, 1, 8, 8000), false), ErrorKind::Unsupported); // mu-law
    assert_eq!(kind(&header(1, 1, 16, 0), false), ErrorKind::Parse);
    assert_eq!(kind(b"RIFX\0\0\0\0WAVE", false), ErrorKind::Parse);
    assert_eq!(kind(b"RIFF\0\0\0\0WAVX", false), ErrorKind::Parse);
    assert_eq!(kind(b"RIFF", false), ErrorKind::Parse); // truncated
    assert_eq!(kind(&header(1, 1, 16, 8000)[..30], false), ErrorKind::Parse);
    // data before fmt
    let mut d = Vec::new();
    d.extend_from_slice(b"RIFF\0\0\0\0WAVEdata\x04\0\0\0\0\0\0\0");
    assert_eq!(kind(&d, false), ErrorKind::Parse);
    // no data chunk at all
    let mut n = header(1, 1, 16, 8000);
    n.truncate(36);
    assert_eq!(kind(&n, false), ErrorKind::Parse);
    // an absurd fmt size
    let mut f = Vec::new();
    f.extend_from_slice(b"RIFF\0\0\0\0WAVEfmt \xff\xff\xff\x7f");
    assert_eq!(kind(&f, false), ErrorKind::Parse);
}

#[test]
fn wav_reader_handles_streamed_and_truncated_data() {
    let mut w = WavWriter::new(Cursor::new(Vec::new()), formats::RS16, rate(8000)).unwrap();
    w.write(&pattern(40), &BlockMeta::default()).unwrap();
    let mut file = w.finish().unwrap().into_inner();
    // data size 0xFFFFFFFF (streamed): reads to the end of the input
    file[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut r = WavReader::new(&file[..], false, 100).unwrap();
    assert_eq!(read_all(&mut r, 1000).0.len(), 40);
    // a size larger than the data: reads what is there
    file[40..44].copy_from_slice(&1000u32.to_le_bytes());
    let mut r = WavReader::new(&file[..], false, 100).unwrap();
    assert_eq!(read_all(&mut r, 1000).0.len(), 40);
    // a size smaller than the data: stops at the size
    file[40..44].copy_from_slice(&10u32.to_le_bytes());
    let mut r = WavReader::new(&file[..], false, 100).unwrap();
    assert_eq!(read_all(&mut r, 1000).0.len(), 10);
}

#[test]
fn wav_writer_validates_inputs() {
    let c = || Cursor::new(Vec::new());
    for format in [
        formats::CU8,
        formats::CS8,
        formats::CS12_LSB16,
        formats::PACKED12,
    ] {
        assert_eq!(
            WavWriter::new(c(), format, rate(8000))
                .err()
                .unwrap()
                .kind(),
            ErrorKind::Unsupported
        );
    }
    assert!(WavWriter::new(c(), formats::RS16, Rate::new(1, 2).unwrap()).is_err());
    assert!(WavWriter::new(c(), formats::RS16, rate(u64::from(u32::MAX) + 1)).is_err());
    let mut w = WavWriter::new(c(), formats::CS16, rate(8000)).unwrap();
    assert!(w.write(&[0; 3], &BlockMeta::default()).is_err());
    let burst = BlockMeta {
        flags: BlockFlags::BURST_START,
        ..BlockMeta::default()
    };
    w.write(&[0; 4], &burst).unwrap();
    assert_eq!(w.flag_counts().burst_start, 1);
    assert_eq!(w.poll_status().unwrap(), None);
    // an empty file is valid
    let w = WavWriter::new(c(), formats::RF32, rate(8000)).unwrap();
    let file = w.finish().unwrap().into_inner();
    assert_eq!(file.len(), 44);
    let mut r = WavReader::new(&file[..], false, 10).unwrap();
    assert_eq!(r.read(&mut [0u8; 8]).unwrap().0, 0);
}

// -------------------------------------------------------------------------------- conformance

#[test]
fn readers_pass_the_rx_conformance_check() {
    // raw
    let data = pattern(2 * 5000);
    let mut raw = RawReader::new(
        &data[..],
        formats::CU8,
        rate(2_400_000),
        SampleIndex(7),
        300,
    )
    .unwrap();
    let report = conformance::check_rx(&mut raw, 100, 512).unwrap();
    assert_eq!(report.blocks, 20); // 5000 samples in chunks of 256 (buffer limited)
    assert!(report.ended);

    // SigMF with a gap, an anchor and a retune
    let mut w = SigmfWriter::new(Vec::new(), formats::CS16, rate(48_000)).unwrap();
    w.set_frequency(1e6).unwrap();
    let anchor = TimeAnchor {
        index: SampleIndex(0),
        unix_ns: 1_000_000_000,
        source: TimeSource::Host,
    };
    w.write(
        &pattern(4 * 300),
        &BlockMeta {
            anchor: Some(anchor),
            ..BlockMeta::default()
        },
    )
    .unwrap();
    w.write(
        &pattern(4 * 200),
        &BlockMeta {
            start: SampleIndex(500),
            ..BlockMeta::default()
        },
    )
    .unwrap();
    w.set_frequency(2e6).unwrap();
    w.write(
        &pattern(4 * 100),
        &BlockMeta {
            start: SampleIndex(700),
            ..BlockMeta::default()
        },
    )
    .unwrap();
    let (data, text) = w.finish().unwrap();
    let mut r = SigmfReader::new(&text, &data[..]).unwrap();
    let report = conformance::check_rx(&mut r, 50, 400).unwrap();
    assert_eq!((report.gaps, report.anchors, report.retunes), (1, 1, 1));
    assert!(report.ended);

    // WAV
    let mut w = WavWriter::new(Cursor::new(Vec::new()), formats::CF32, rate(48_000)).unwrap();
    w.write(&pattern(8 * 1000), &BlockMeta::default()).unwrap();
    let file = w.finish().unwrap().into_inner();
    let mut r = WavReader::new(&file[..], true, 128).unwrap();
    let report = conformance::check_rx(&mut r, 100, 2048).unwrap();
    assert_eq!(report.samples, 1000);
}

#[test]
fn writers_pass_the_tx_conformance_check() {
    let mut raw = RawWriter::new(Vec::new(), formats::CS16, rate(48_000));
    let report = conformance::check_tx(&mut raw, 5, 64).unwrap();
    assert_eq!((report.blocks, report.samples), (5, 80));
    assert_eq!(raw.flag_counts().burst_start, 1);

    let mut sigmf = SigmfWriter::new(Vec::new(), formats::CF32, rate(48_000)).unwrap();
    assert_eq!(
        conformance::check_tx(&mut sigmf, 5, 64).unwrap().samples,
        40
    );

    let mut wav = WavWriter::new(Cursor::new(Vec::new()), formats::RS16, rate(48_000)).unwrap();
    assert_eq!(conformance::check_tx(&mut wav, 5, 64).unwrap().samples, 160);
}
