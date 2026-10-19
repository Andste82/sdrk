//! A recording written as SigMF (with a gap and annotations), read back and pushed through
//! `fm_receiver.json` gives the same audio and events as pushing the blocks directly.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
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
    clippy::unreadable_literal
)]

use sdrk_device::{
    BlockFlags, BlockMeta, Rate, RxStream, SampleIndex, TimeAnchor, TimeSource, TxStream, formats,
};
use sdrk_io::{Annotation, SigmfReader, SigmfWriter};
use sdrk_pipeline::{Event, InputData, Pipeline};
use sdrk_system_tests::{AUDIO_RATE, IF_RATE, build, drain_events, encode_bytes, flow, fm_station};
use sdrk_testkit::meters::sinad_db;

const CHUNK: usize = 4096;
const T0_NS: i64 = 1_700_000_000_000_000_000;

/// Pushes blocks and collects audio bits and events.
fn run(
    p: &mut Pipeline,
    blocks: impl Iterator<Item = (Vec<u8>, BlockMeta)>,
) -> (Vec<u32>, Vec<Event>) {
    let mut audio = Vec::new();
    let mut events = Vec::new();
    let mut buf = vec![0f32; 4096];
    let mut frame = vec![0f32; 1024];
    for (bytes, meta) in blocks {
        p.push_with_meta("rf", InputData::Raw(&bytes), &meta)
            .unwrap();
        loop {
            let n = p.read_stream::<f32>("audio", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            audio.extend(buf[..n].iter().map(|v| v.to_bits()));
        }
        while p.read_frame("spectrum", &mut frame).unwrap().is_some() {}
        drain_events(p, &mut events);
    }
    (audio, events)
}

#[test]
fn sigmf_roundtrip_pipeline() {
    // The recording: 0.5 s of the E2E-RX signal in CU8, in three segments with a gap of 7777
    // samples before the second one and an anchor on the first block.
    let total = 1_200_000usize;
    let iq = fm_station(600_000.0, 1000.0, 0.9, 0, total);
    let bytes = encode_bytes(&formats::CU8, &iq);
    let gap = 7777u64;
    let segments = [
        (0usize, 400_000usize),
        (400_000, 400_000),
        (800_000, 400_000),
    ];
    // Segment 1 follows segment 0 contiguously, segment 2 follows after the gap.
    let start_of = |k: usize| match k {
        0 => 0u64,
        1 => 400_000,
        _ => 800_000 + gap,
    };

    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("recording");
    let rate = Rate::hz(IF_RATE).unwrap();
    let mut w = SigmfWriter::create(&base, formats::CU8, rate).unwrap();
    w.set_frequency(100e6).unwrap();
    let anchor = TimeAnchor {
        index: SampleIndex(0),
        unix_ns: T0_NS,
        source: TimeSource::Host,
    };
    for (k, &(from, n)) in segments.iter().enumerate() {
        let meta = BlockMeta {
            start: SampleIndex(start_of(k)),
            anchor: (k == 0).then_some(anchor),
            ..BlockMeta::default()
        };
        // several writes per segment: they must stay one capture
        let piece = n / 4;
        for q in 0..4 {
            let m = if q == 0 {
                meta
            } else {
                BlockMeta {
                    start: SampleIndex(start_of(k) + (q * piece) as u64),
                    ..BlockMeta::default()
                }
            };
            let lo = (from + q * piece) * 2;
            w.write(&bytes[lo..lo + piece * 2], &m).unwrap();
        }
    }
    let annotations = vec![
        Annotation {
            sample_start: 1000,
            sample_count: 5000,
            freq_lower_edge: Some(100e6 + 500e3),
            freq_upper_edge: Some(100e6 + 700e3),
            label: Some("station".into()),
            comment: None,
        },
        Annotation {
            sample_start: 399_000,
            sample_count: 2000,
            comment: Some("across the first boundary".into()),
            ..Annotation::default()
        },
        Annotation {
            sample_start: 1_100_000,
            sample_count: 10,
            label: Some("end".into()),
            ..Annotation::default()
        },
    ];
    for a in &annotations {
        w.annotate(a.clone()).unwrap();
    }
    w.finish().unwrap();

    // Read back
    let mut reader = SigmfReader::open(&base)
        .unwrap()
        .with_chunk_samples(CHUNK)
        .unwrap();
    assert_eq!(reader.annotations(), &annotations[..]);
    assert_eq!(
        reader.captures().len(),
        2,
        "the gap starts the second capture"
    );
    assert_eq!(reader.format(), formats::CU8);
    assert_eq!(reader.sample_rate(), rate);
    let mut from_file = Vec::new();
    let mut buf = vec![0u8; CHUNK * 2];
    loop {
        let (n, meta) = reader.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        from_file.push((buf[..n].to_vec(), meta));
    }

    // Direct: the same blocks as the writer was given, cut into pieces of CHUNK samples.
    let mut direct = Vec::new();
    for (start, count, flags, anchor_here) in [
        (0u64, 800_000usize, BlockFlags::empty(), true),
        (800_000 + gap, 400_000, BlockFlags::GAP_BEFORE, false),
    ] {
        let first_sample = if start == 0 { 0 } else { 800_000 };
        let mut done = 0usize;
        while done < count {
            let n = CHUNK.min(count - done);
            let lo = (first_sample + done) * 2;
            let meta = BlockMeta {
                start: SampleIndex(start + done as u64),
                flags: if done == 0 {
                    flags
                } else {
                    BlockFlags::empty()
                },
                anchor: (done == 0 && anchor_here).then_some(TimeAnchor {
                    index: SampleIndex(0),
                    unix_ns: T0_NS,
                    source: TimeSource::Host,
                }),
                tx_time_ns: None,
            };
            direct.push((bytes[lo..lo + n * 2].to_vec(), meta));
            done += n;
        }
    }
    assert_eq!(from_file.len(), direct.len());
    for (k, (a, b)) in from_file.iter().zip(&direct).enumerate() {
        assert_eq!(a.0, b.0, "bytes of block {k}");
        assert_eq!(a.1, b.1, "metadata of block {k}");
    }

    let spec = flow("fm_receiver");
    let mut p_file = build(&spec);
    let mut p_direct = build(&spec);
    let (audio_file, events_file) = run(&mut p_file, from_file.into_iter());
    let (audio_direct, events_direct) = run(&mut p_direct, direct.into_iter());
    assert!(audio_file.len() > 20_000);
    assert_eq!(audio_file, audio_direct, "audio is bit-identical");
    assert_eq!(events_file, events_direct, "events are identical");
    // one anchor and one gap on each of the two outputs
    assert_eq!(events_file.len(), 4);

    // Audio quality of the first (gap-free) part.
    let first: Vec<f64> = audio_file[8000..16000]
        .iter()
        .map(|b| f64::from(f32::from_bits(*b)))
        .collect();
    assert!(sinad_db(&first, AUDIO_RATE, 1000.0) > 40.0);
}
