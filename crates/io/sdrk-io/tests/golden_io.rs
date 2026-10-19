//! Golden I/O vectors: SigMF metadata, WAV files and ISO 8601 timestamps produced by Python.
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
    BlockFlags, BlockMeta, Rate, RxStream, SampleIndex, TimeAnchor, TimeSource, TxStream, formats,
};
use sdrk_io::{
    Annotation, SigmfReader, SigmfWriter, WavReader, WavWriter, format_iso8601_ns, parse_iso8601_ns,
};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use serde_json::Value;

fn bytes_of(case: &GoldenCase, name: &str) -> Vec<u8> {
    match case.array(name).unwrap() {
        Array::U8(v) => v.clone(),
        other => panic!("array {name} is {}, not u8", other.dtype()),
    }
}

fn flags_of(names: &[Value]) -> BlockFlags {
    let mut f = BlockFlags::empty();
    for n in names {
        f.insert(match n.as_str().unwrap() {
            "retuned" => BlockFlags::RETUNED,
            "gap_before" => BlockFlags::GAP_BEFORE,
            "overflow" => BlockFlags::OVERFLOW,
            other => panic!("unknown flag {other}"),
        });
    }
    f
}

/// What a step of the case does, for the writer and for the reader check.
struct Written {
    meta: BlockMeta,
    samples: u64,
}

fn run_sigmf_case(name: &str) {
    let case = golden::load("io", name).unwrap();
    let p = &case.params;
    let datatype = p["datatype"].as_str().unwrap();
    let format = sdrk_io::format_of_datatype(datatype).unwrap();
    let rate = Rate::new(
        p["rate_num"].as_u64().unwrap(),
        p["rate_den"].as_u64().unwrap(),
    )
    .unwrap();
    let sample_bytes = format.bytes_for_samples(1).unwrap();

    let mut w = SigmfWriter::new(Vec::new(), format, rate).unwrap();
    if let Some(d) = p.get("description") {
        w.set_description(d.as_str().unwrap());
    }
    let mut written = Vec::new();
    let mut frequencies = Vec::new();
    let mut frequency = None;
    let mut total_bytes = 0usize;
    for step in p["steps"].as_array().unwrap() {
        match step["op"].as_str().unwrap() {
            "frequency" => {
                let hz = step["hz"].as_f64().unwrap();
                w.set_frequency(hz).unwrap();
                frequency = Some(hz);
            }
            "write" => {
                let start = step["start"].as_u64().unwrap();
                let samples = step["samples"].as_u64().unwrap();
                let anchor = step.get("anchor").map(|a| TimeAnchor {
                    index: SampleIndex(a["index"].as_u64().unwrap()),
                    unix_ns: a["unix_ns"].as_i64().unwrap(),
                    source: TimeSource::Host,
                });
                let flags = step
                    .get("flags")
                    .map_or(BlockFlags::empty(), |f| flags_of(f.as_array().unwrap()));
                let meta = BlockMeta {
                    start: SampleIndex(start),
                    flags,
                    anchor,
                    tx_time_ns: None,
                };
                // payload: a counter pattern so that the reader check can verify the bytes
                let payload: Vec<u8> = (0..samples as usize * sample_bytes)
                    .map(|i| ((i + total_bytes) % 251) as u8)
                    .collect();
                total_bytes += payload.len();
                assert_eq!(w.write(&payload, &meta).unwrap(), payload.len());
                written.push(Written { meta, samples });
                frequencies.push(frequency);
            }
            other => panic!("unknown op {other}"),
        }
    }
    for a in p["annotations"].as_array().unwrap() {
        w.annotate(Annotation {
            sample_start: a["sample_start"].as_u64().unwrap(),
            sample_count: a["sample_count"].as_u64().unwrap(),
            freq_lower_edge: a.get("freq_lower_edge").map(|v| v.as_f64().unwrap()),
            freq_upper_edge: a.get("freq_upper_edge").map(|v| v.as_f64().unwrap()),
            label: a.get("label").map(|v| v.as_str().unwrap().to_string()),
            comment: a.get("comment").map(|v| v.as_str().unwrap().to_string()),
        })
        .unwrap();
    }
    let (data, meta_text) = w.finish().unwrap();
    let expected = bytes_of(&case, "expected");
    assert_eq!(
        meta_text.as_bytes(),
        &expected[..],
        "{name}: writer output differs from the golden file\n{meta_text}"
    );

    // Group the writes into capture segments the way the format defines them.
    struct Segment {
        start: u64,
        samples: u64,
        gap: bool,
        retuned: bool,
        anchor: Option<TimeAnchor>,
    }
    let mut segments: Vec<Segment> = Vec::new();
    let mut end: Option<u64> = None;
    let mut open_frequency: Option<f64> = None;
    for (wr, frequency) in written.iter().zip(&frequencies) {
        let new = segments.is_empty()
            || end != Some(wr.meta.start.0)
            || wr.meta.flags.contains(BlockFlags::RETUNED)
            || wr.meta.anchor.is_some()
            || *frequency != open_frequency;
        if new {
            segments.push(Segment {
                start: wr.meta.start.0,
                samples: 0,
                gap: end.is_some_and(|e| wr.meta.start.0 > e),
                retuned: open_frequency.is_some()
                    && frequency.is_some()
                    && *frequency != open_frequency,
                anchor: wr.meta.anchor,
            });
            open_frequency = *frequency;
        }
        segments.last_mut().unwrap().samples += wr.samples;
        end = Some(wr.meta.start.0 + wr.samples);
    }

    // The reader parses the golden bytes and returns one block per segment.
    let mut r =
        SigmfReader::new(std::str::from_utf8(&expected).unwrap(), Cursor::new(data)).unwrap();
    assert_eq!(r.format(), format);
    assert_eq!(r.sample_rate(), rate);
    assert_eq!(r.captures().len(), segments.len());
    assert_eq!(
        r.annotations().len(),
        p["annotations"].as_array().unwrap().len()
    );
    let mut buf = vec![0u8; 100_000 * sample_bytes];
    let mut payload_position = 0usize;
    for (k, seg) in segments.iter().enumerate() {
        let (n, meta) = r.read(&mut buf).unwrap();
        assert_eq!(
            n as u64,
            seg.samples * sample_bytes as u64,
            "{name}: segment {k} size"
        );
        assert_eq!(meta.start.0, seg.start, "{name}: segment {k} start");
        assert_eq!(
            meta.flags.contains(BlockFlags::GAP_BEFORE),
            seg.gap,
            "{name}: gap of segment {k}"
        );
        assert_eq!(
            meta.flags.contains(BlockFlags::RETUNED),
            seg.retuned,
            "{name}: retune of segment {k}"
        );
        match (meta.anchor, seg.anchor) {
            (None, None) => {}
            (Some(got), Some(want)) => {
                assert_eq!((got.index.0, got.unix_ns), (seg.start, want.unix_ns));
                assert_eq!(got.source, TimeSource::Host);
            }
            (got, want) => panic!("{name}: anchor of segment {k}: {got:?} vs {want:?}"),
        }
        for (i, b) in buf[..n].iter().enumerate() {
            assert_eq!(*b, ((i + payload_position) % 251) as u8);
        }
        payload_position += n;
    }
    assert_eq!(r.read(&mut buf).unwrap().0, 0);
}

#[test]
fn sigmf_cu8_meta_matches_python() {
    run_sigmf_case("sigmf_cu8_meta");
}

#[test]
fn sigmf_fractional_rate_meta_matches_python() {
    run_sigmf_case("sigmf_cf32_fractional_meta");
}

#[test]
fn sigmf_reader_exposes_captures_and_annotations_of_the_golden_file() {
    let case = golden::load("io", "sigmf_cu8_meta").unwrap();
    let text = String::from_utf8(bytes_of(&case, "expected")).unwrap();
    let r = SigmfReader::new(&text, Cursor::new(Vec::new())).unwrap();
    let caps = r.captures();
    assert_eq!(caps.len(), 5);
    assert_eq!(caps[0].datetime_ns, Some(1_700_000_000_123_456_789));
    assert_eq!(caps[1].global_index, 1700);
    assert_eq!(caps[2].frequency_hz, Some(101_000_000.0));
    assert_eq!(caps[4].datetime_ns, Some(1_700_000_001_000_000_000));
    assert_eq!(r.description(), Some("gap, retune and anchor test"));
    let a = r.annotations();
    assert_eq!(a.len(), 3);
    assert_eq!(a[0].label.as_deref(), Some("carrier"));
    assert_eq!(a[0].freq_lower_edge, Some(-50_000.0));
    assert_eq!(a[1].comment.as_deref(), Some("burst after the gap"));
    assert_eq!((a[2].sample_start, a[2].sample_count), (1900, 10));
}

fn wav_writer_case(name: &str) {
    let case = golden::load("io", name).unwrap();
    let p = &case.params;
    let format = match p["format"].as_str().unwrap() {
        "rs16" => formats::RS16,
        "cf32" => formats::CF32,
        "cs16" => formats::CS16,
        "rf32" => formats::RF32,
        other => panic!("format {other}"),
    };
    let rate = Rate::hz(p["rate"].as_u64().unwrap()).unwrap();
    let payload: Vec<u8> = match case.array("samples").unwrap() {
        Array::I16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Array::CF32(v) => v
            .iter()
            .flat_map(|c| [c.re.to_le_bytes(), c.im.to_le_bytes()].concat())
            .collect(),
        other => panic!("samples are {}", other.dtype()),
    };
    let mut w = WavWriter::new(Cursor::new(Vec::new()), format, rate).unwrap();
    // two pieces, to exercise appending
    let half = payload.len() / 2;
    w.write(&payload[..half], &BlockMeta::default()).unwrap();
    w.write(&payload[half..], &BlockMeta::default()).unwrap();
    let got = w.finish().unwrap().into_inner();
    assert_eq!(got, bytes_of(&case, "expected"), "{name}");

    // and the reader parses the Python file
    let iq = p["iq"].as_bool().unwrap();
    let mut r = WavReader::new(&got[..], iq, 1000).unwrap();
    assert_eq!(r.format(), format);
    assert_eq!(r.sample_rate(), rate);
    let mut buf = vec![0u8; payload.len() + 16];
    let (n, meta) = r.read(&mut buf).unwrap();
    assert_eq!(&buf[..n], &payload[..]);
    assert_eq!(meta.start.0, 0);
    assert_eq!(r.read(&mut buf).unwrap().0, 0);
}

#[test]
fn wav_pcm16_mono_matches_python() {
    wav_writer_case("wav_pcm16_mono");
}

#[test]
fn wav_float32_stereo_matches_python() {
    wav_writer_case("wav_float32_stereo");
}

fn wav_reader_case(name: &str, format: sdrk_device::SampleFormat) {
    let case = golden::load("io", name).unwrap();
    let file = bytes_of(&case, "file");
    let payload: Vec<u8> = match case.array("samples").unwrap() {
        Array::CI16(v) => v
            .iter()
            .flat_map(|c| [c.re.0.to_le_bytes(), c.im.0.to_le_bytes()].concat())
            .collect(),
        Array::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        other => panic!("samples are {}", other.dtype()),
    };
    let iq = case.params["iq"].as_bool().unwrap();
    let mut r = WavReader::new(&file[..], iq, 7).unwrap();
    assert_eq!(r.format(), format);
    assert_eq!(
        r.sample_rate(),
        Rate::hz(case.params["rate"].as_u64().unwrap()).unwrap()
    );
    let mut out = Vec::new();
    let mut buf = vec![0u8; 1024];
    loop {
        let (n, _) = r.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        assert!(n <= 7 * format.bytes_for_samples(1).unwrap());
        out.extend_from_slice(&buf[..n]);
    }
    assert_eq!(out, payload, "{name}");
}

#[test]
fn wav_reader_parses_extensible_and_skips_the_list_chunk() {
    wav_reader_case("wav_extensible_cs16_list", formats::CS16);
}

#[test]
fn wav_reader_parses_extensible_float_mono() {
    wav_reader_case("wav_extensible_rf32_mono", formats::RF32);
}

#[test]
fn iso8601_table_matches_python_datetime() {
    let case = golden::load("io", "iso8601_table").unwrap();
    let entries = case.params["cases"].as_array().unwrap();
    assert_eq!(entries.len(), 30);
    for e in entries {
        let text = e["text"].as_str().unwrap();
        let ns = e["unix_ns"].as_i64().unwrap();
        assert_eq!(parse_iso8601_ns(text).unwrap(), ns, "{text}");
        // formatting gives back an equivalent string that parses to the same value
        let formatted = format_iso8601_ns(ns);
        assert_eq!(
            parse_iso8601_ns(&formatted).unwrap(),
            ns,
            "{text} -> {formatted}"
        );
    }
    for bad in case.params["invalid"].as_array().unwrap() {
        let text = bad.as_str().unwrap();
        assert!(parse_iso8601_ns(text).is_err(), "`{text}` must be rejected");
    }
}

#[test]
fn iso8601_format_is_canonical() {
    let cases = [
        (0, "1970-01-01T00:00:00Z"),
        (1, "1970-01-01T00:00:00.000000001Z"),
        (-1, "1969-12-31T23:59:59.999999999Z"),
        (1_500_000_000, "1970-01-01T00:00:01.5Z"),
        (951_825_600_500_000_000, "2000-02-29T12:00:00.5Z"),
        (1_700_000_000_123_456_789, "2023-11-14T22:13:20.123456789Z"),
        (i64::MAX, "2262-04-11T23:47:16.854775807Z"),
    ];
    for (ns, text) in cases {
        assert_eq!(format_iso8601_ns(ns), text);
        assert_eq!(parse_iso8601_ns(text).unwrap(), ns);
    }
}
