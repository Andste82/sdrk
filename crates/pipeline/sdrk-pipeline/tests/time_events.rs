//! Index domains, gaps, anchors, joining mid-stream, events and timed parameters.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::assert_is_empty,
    clippy::unreadable_literal,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::redundant_closure_for_method_calls
)]

mod common;

use sdrk::C32;
use sdrk_blocks::{Block, DdcBlock, DdcConfig, FrameMeta, ParamValue};
use sdrk_device::{BlockFlags, BlockMeta, SampleIndex, TimeAnchor, TimeSource};
use sdrk_pipeline::{
    BuildOptions, Error, Event, EventKind, InputData, ParamTime, Pipeline, PipelineSpec, Registry,
};
use serde_json::{Value, json};

use common::{bits, build, cu8_recording, flow};

/// What a run produced.
#[derive(Default, Debug, Clone, PartialEq)]
struct Run {
    audio: Vec<f32>,
    frames: Vec<(Vec<f32>, FrameMeta)>,
}

/// Pushes `bytes` (CU8) in blocks of `chunk` samples with metadata that start at the sample
/// index `start`, and collects audio and frames after every push.
fn feed_meta(p: &mut Pipeline, bytes: &[u8], start: u64, chunk: usize, run: &mut Run) {
    let mut buf = vec![0f32; 4096];
    let mut frame = vec![0f32; 1024];
    let mut index = start;
    for block in bytes.chunks(2 * chunk) {
        let meta = BlockMeta {
            start: SampleIndex(index),
            ..BlockMeta::default()
        };
        p.push_with_meta("rf", InputData::Raw(block), &meta)
            .unwrap();
        index += (block.len() / 2) as u64;
        loop {
            let n = p.read_stream("audio", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            run.audio.extend_from_slice(&buf[..n]);
        }
        while let Some(m) = p.read_frame("spectrum", &mut frame).unwrap() {
            run.frames.push((frame.clone(), m));
        }
    }
}

fn events(p: &mut Pipeline) -> Vec<Event> {
    std::iter::from_fn(|| p.poll_event()).collect()
}

#[test]
fn a_gap_equals_two_separately_reset_runs() {
    const N1: usize = 40_000;
    const GAP: u64 = 12_345;
    const N2: usize = 30_000;
    let all = cu8_recording(N1 + GAP as usize + N2);
    let seg1 = &all[..2 * N1];
    let seg2 = &all[2 * (N1 + GAP as usize)..];

    let mut gapped = build(&flow("fm_receiver"));
    let mut run = Run::default();
    feed_meta(&mut gapped, seg1, 0, 3000, &mut run);
    let end1 = gapped.read_index("audio").unwrap(); // everything was read
    assert_eq!(gapped.available("audio").unwrap(), 0);
    let audio_end1 = run.audio.len() as u64;
    assert_eq!(end1, audio_end1);
    let first_events = events(&mut gapped);
    assert!(
        first_events.is_empty(),
        "a first block fixes the origin: no event"
    );
    let frames_end1 = run.frames.len();
    feed_meta(&mut gapped, seg2, (N1 as u64) + GAP, 3000, &mut run);

    let mut one = build(&flow("fm_receiver"));
    let mut run1 = Run::default();
    feed_meta(&mut one, seg1, 0, 3000, &mut run1);
    let mut two = build(&flow("fm_receiver"));
    let mut run2 = Run::default();
    feed_meta(&mut two, seg2, 0, 3000, &mut run2);

    // The data equals the two separate runs.
    let mut joined = run1.audio.clone();
    joined.extend_from_slice(&run2.audio);
    assert_eq!(bits(&run.audio), bits(&joined), "audio");
    assert_eq!(run.frames.len(), run1.frames.len() + run2.frames.len());
    for (a, b) in run.frames[..frames_end1].iter().zip(&run1.frames) {
        assert_eq!(bits(&a.0), bits(&b.0));
        assert_eq!(a.1, b.1);
    }
    // The frames after the gap: same values, first_sample moved by the gap.
    for (a, b) in run.frames[frames_end1..].iter().zip(&run2.frames) {
        assert_eq!(bits(&a.0), bits(&b.0));
        assert_eq!(a.1.first_sample, b.1.first_sample + N1 as u64 + GAP);
        assert_eq!(a.1.averaged, b.1.averaged);
    }

    // The Gap events: the missing samples at the input, at the mapped index of every output.
    let ev = events(&mut gapped);
    let audio = gapped.output_id("audio").unwrap();
    let spectrum = gapped.output_id("spectrum").unwrap();
    assert_eq!(ev.len(), 2, "{ev:?}");
    // ddc divides by 10, the demodulator by 5: 12 345 -> 1 235 -> 247 audio samples.
    let audio_gap = ev.iter().find(|e| e.output == audio).unwrap();
    assert_eq!(audio_gap.kind, EventKind::Gap { missing: GAP });
    assert_eq!(audio_gap.index, audio_end1 + 247);
    let spectrum_gap = ev.iter().find(|e| e.output == spectrum).unwrap();
    assert_eq!(spectrum_gap.kind, EventKind::Gap { missing: GAP });
    assert_eq!(spectrum_gap.index, N1 as u64 + GAP);
    // The index of the first sample after the gap is the index of the event.
    let mut again = build(&flow("fm_receiver"));
    let mut sink = Run::default();
    feed_meta(&mut again, seg1, 0, 3000, &mut sink);
    let before = again.read_index("audio").unwrap();
    assert_eq!(before, audio_end1);
    let meta = BlockMeta {
        start: SampleIndex(N1 as u64 + GAP),
        ..BlockMeta::default()
    };
    again
        .push_with_meta("rf", InputData::Raw(&seg2[..2 * 3000]), &meta)
        .unwrap();
    assert_eq!(again.read_index("audio").unwrap(), audio_end1 + 247);
}

#[test]
fn time_cannot_go_backwards() {
    let mut p = build(&flow("fm_receiver"));
    let bytes = cu8_recording(2000);
    let meta = BlockMeta {
        start: SampleIndex(100),
        ..BlockMeta::default()
    };
    p.push_with_meta("rf", InputData::Raw(&bytes[..2000]), &meta)
        .unwrap();
    let back = BlockMeta {
        start: SampleIndex(1000),
        ..BlockMeta::default()
    };
    let err = p
        .push_with_meta("rf", InputData::Raw(&bytes[..2000]), &back)
        .unwrap_err();
    assert!(matches!(err, Error::InvalidState(_)), "{err}");
    // The exactly expected start is fine.
    let ok = BlockMeta {
        start: SampleIndex(1100),
        ..BlockMeta::default()
    };
    p.push_with_meta("rf", InputData::Raw(&bytes[..2000]), &ok)
        .unwrap();
    assert_eq!(p.input_index("rf").unwrap(), 2100);
    // Raw data must be whole samples.
    let odd = BlockMeta {
        start: SampleIndex(2100),
        ..BlockMeta::default()
    };
    assert!(matches!(
        p.push_with_meta("rf", InputData::Raw(&bytes[..3]), &odd),
        Err(Error::LengthMismatch { .. })
    ));
}

#[test]
fn joining_mid_stream_gives_the_same_results() {
    let all = cu8_recording(60_000);
    let mut a = build(&flow("fm_receiver"));
    let mut ra = Run::default();
    feed_meta(&mut a, &all, 0, 3000, &mut ra);
    let mut b = build(&flow("fm_receiver"));
    let mut rb = Run::default();
    let start = 1_000_000_000u64;
    feed_meta(&mut b, &all, start, 3000, &mut rb);
    assert_eq!(bits(&ra.audio), bits(&rb.audio));
    assert_eq!(ra.frames.len(), rb.frames.len());
    for (x, y) in ra.frames.iter().zip(&rb.frames) {
        assert_eq!(bits(&x.0), bits(&y.0));
        assert_eq!(y.1.first_sample, x.1.first_sample + start);
    }
    // Joining is not a gap: no event.
    assert!(events(&mut b).is_empty());
    // The output index starts at the mapped start: 10^9 / 10 / 5 = 20 000 000.
    assert_eq!(b.input_index("rf").unwrap(), start + 60_000);
    assert_eq!(
        b.read_index("audio").unwrap(),
        20_000_000 + rb.audio.len() as u64
    );
}

/// The `fm_receiver` flow without de-emphasis, so that the only delay is the group delay.
fn fm_no_deemphasis() -> String {
    let mut v: Value = serde_json::from_str(&flow("fm_receiver")).unwrap();
    v["nodes"]["demod"]["params"]["deemphasis_us"] = json!(0);
    v.to_string()
}

#[test]
fn an_anchor_on_the_input_gives_the_time_of_the_signal_on_the_output() {
    // The tone of the audio is known analytically: audio(t) = sin(2 pi f t) with
    // t = n / 2.4 MHz counted from input sample 0, which has the absolute time T0.
    const T0: i64 = 1_700_000_000_000_000_000;
    let samples = 400_000;
    let bytes = cu8_recording(samples);
    let mut p = build(&fm_no_deemphasis());
    let mut audio = Vec::new();
    let mut buf = vec![0f32; 4096];
    let mut first = None;
    let mut index = 0u64;
    for block in bytes.chunks(2 * 3000) {
        let anchor = (index == 0).then_some(TimeAnchor {
            index: SampleIndex(0),
            unix_ns: T0,
            source: TimeSource::Host,
        });
        let meta = BlockMeta {
            start: SampleIndex(index),
            anchor,
            ..BlockMeta::default()
        };
        p.push_with_meta("rf", InputData::Raw(block), &meta)
            .unwrap();
        index += (block.len() / 2) as u64;
        loop {
            let n = p.read_stream("audio", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            if first.is_none() {
                first = Some(p.read_index("audio").unwrap() - n as u64);
            }
            audio.extend_from_slice(&buf[..n]);
        }
    }
    let first = first.unwrap();
    assert_eq!(first, 0);
    // One Anchor event per output, delay compensated: it carries the mapped index.
    let ev = events(&mut p);
    let a = p.output_id("audio").unwrap();
    let anchor_events: Vec<_> = ev
        .iter()
        .filter(|e| matches!(e.kind, EventKind::Anchor(_)))
        .collect();
    assert_eq!(anchor_events.len(), 2, "{ev:?}");
    let audio_anchor = anchor_events.iter().find(|e| e.output == a).unwrap();
    let EventKind::Anchor(at) = audio_anchor.kind else {
        unreachable!()
    };
    assert_eq!(at.index.0, audio_anchor.index);
    // Fit y = A sin(w t) + B cos(w t) over the settled part; the phase is the time error.
    let w = 2.0 * std::f64::consts::PI * 1000.0;
    let (mut ss, mut cc, mut sc, mut ys, mut yc) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for m in 500..audio.len() {
        let t_ns = p.time_of("audio", m as u64).unwrap().unwrap();
        let t = (t_ns - T0) as f64 * 1e-9;
        let (s, c) = (w * t).sin_cos();
        let y = f64::from(audio[m]);
        ss += s * s;
        cc += c * c;
        sc += s * c;
        ys += y * s;
        yc += y * c;
    }
    let det = ss * cc - sc * sc;
    let amp_s = (ys * cc - yc * sc) / det;
    let amp_c = (yc * ss - ys * sc) / det;
    assert!(amp_s > 0.5, "the tone is found: {amp_s}");
    let phase = amp_c.atan2(amp_s);
    let error_s = phase / w;
    let half_sample = 0.5 / 48_000.0;
    assert!(
        error_s.abs() < half_sample,
        "time error {error_s} s, half an audio sample is {half_sample} s"
    );
}

#[test]
fn flags_become_events_at_the_mapped_positions_of_every_output() {
    let mut p = build(&flow("fm_receiver"));
    let bytes = cu8_recording(20_000);
    let flags = BlockFlags::OVERFLOW
        .union(BlockFlags::RETUNED)
        .union(BlockFlags::LATE)
        .union(BlockFlags::BURST_START)
        .union(BlockFlags::BURST_END);
    let meta = BlockMeta {
        start: SampleIndex(1000),
        flags,
        tx_time_ns: Some(42),
        ..BlockMeta::default()
    };
    p.push_with_meta("rf", InputData::Raw(&bytes[..2 * 3000]), &meta)
        .unwrap();
    let ev = events(&mut p);
    let audio = p.output_id("audio").unwrap();
    let spectrum = p.output_id("spectrum").unwrap();
    let on = |o| ev.iter().filter(|e| e.output == o).collect::<Vec<_>>();
    // Five events per output, one per flag.
    assert_eq!(on(audio).len(), 5, "{ev:?}");
    assert_eq!(on(spectrum).len(), 5);
    // The start of the block (sample 1000) maps to 1000/10/5 = 20 on the audio output. The last
    // sample (3999) is 2999 samples after the origin: ceil(2999/10) = 300 after the origin 100,
    // then ceil(300/5) = 60 after the origin 20.
    let audio_events = on(audio);
    let find = |k: fn(&EventKind) -> bool| audio_events.iter().find(|e| k(&e.kind)).unwrap();
    assert_eq!(find(|k| matches!(k, EventKind::Overflow)).index, 20);
    assert_eq!(find(|k| matches!(k, EventKind::Retuned)).index, 20);
    assert_eq!(find(|k| matches!(k, EventKind::Late)).index, 20);
    let start = find(|k| matches!(k, EventKind::BurstStart { .. }));
    assert_eq!(
        start.kind,
        EventKind::BurstStart {
            tx_time_ns: Some(42)
        }
    );
    assert_eq!(start.index, 20);
    assert_eq!(find(|k| matches!(k, EventKind::BurstEnd)).index, 80);
    // On the frames output the index is the input index of the frame node.
    let frame_events = on(spectrum);
    assert!(
        frame_events
            .iter()
            .any(|e| e.kind == EventKind::Overflow && e.index == 1000)
    );
}

#[test]
fn the_event_queue_drops_the_newest_and_reports_the_loss() {
    let mut v: Value = serde_json::from_str(&flow("fm_receiver")).unwrap();
    v["settings"]["event_queue"] = json!(4);
    let mut p = build(&v.to_string());
    let bytes = cu8_recording(1000);
    let mut index = 0u64;
    for _ in 0..5 {
        let meta = BlockMeta {
            start: SampleIndex(index),
            flags: BlockFlags::OVERFLOW,
            ..BlockMeta::default()
        };
        p.push_with_meta("rf", InputData::Raw(&bytes[..200]), &meta)
            .unwrap();
        index += 100;
    }
    // 5 blocks x 2 outputs = 10 events; the queue holds 4.
    let ev = events(&mut p);
    assert_eq!(ev.len(), 5, "{ev:?}");
    assert!(ev[..4].iter().all(|e| e.kind == EventKind::Overflow));
    assert_eq!(ev[4].kind, EventKind::EventsLost { count: 6 });
    // The first lost event is named.
    assert_eq!(ev[4].output, p.output_id("audio").unwrap());
    assert_eq!(p.poll_event(), None);
}

// ------------------------------------------------------------------ timed parameters

/// c32 in, a `ddc` with the given decimation and a stream out.
fn ddc_spec(decimation: u64, extra_settings: Value) -> String {
    json!({
        "sdrk": 1, "name": "timed",
        "settings": extra_settings,
        "inputs": { "iq": { "type": "c32", "rate_hz": 2_400_000 } },
        "nodes": { "tune": { "kind": "ddc", "params": {
            "decimation": decimation, "frequency_hz": 600_000.0, "cutoff_hz": 100_000.0 } } },
        "edges": [ { "from": "iq", "to": "tune" }, { "from": "tune", "to": "bb" } ],
        "outputs": { "bb": { "type": "stream", "capacity": 20000 } }
    })
    .to_string()
}

/// A tone at +600 kHz, complex.
fn tone(n: usize) -> Vec<C32> {
    (0..n)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * 600_000.0 * i as f64 / 2_400_000.0;
            C32::new(phase.cos() as f32, phase.sin() as f32)
        })
        .collect()
}

fn read_bb(p: &mut Pipeline, out: &mut Vec<C32>) {
    let mut buf = vec![C32::default(); 4096];
    loop {
        let n = p.read_stream("bb", &mut buf).unwrap();
        if n == 0 {
            return;
        }
        out.extend_from_slice(&buf[..n]);
    }
}

fn c32_bits(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

/// The reference: the block itself, split by hand at the input sample `at`.
fn manual_split(decimation: usize, input: &[C32], at: usize, new_hz: f64) -> Vec<C32> {
    let cfg = DdcConfig {
        decimation,
        frequency_hz: 600_000.0,
        cutoff_hz: Some(100_000.0),
        ..DdcConfig::default()
    };
    let mut block = DdcBlock::<C32, _>::new(
        &cfg,
        sdrk::Rate::hz(2_400_000).unwrap(),
        &mut sdrk_pipeline::VecStorage,
    )
    .unwrap();
    let mut out = vec![C32::default(); input.len()];
    let mut produced = 0;
    for (k, part) in [&input[..at], &input[at..]].into_iter().enumerate() {
        if k == 1 {
            block
                .set_param("frequency_hz", ParamValue::F64(new_hz))
                .unwrap();
        }
        let mut rest = part;
        while !rest.is_empty() {
            let p = block.process(rest, &mut out[produced..]).unwrap();
            produced += p.produced;
            rest = &rest[p.consumed..];
        }
    }
    out.truncate(produced);
    out
}

fn timed_change(decimation: u64) {
    let n = 40_000usize;
    let at = 12_345u64;
    let input = tone(n);
    let spec = ddc_spec(decimation, json!({ "precision": "f32", "max_chunk": 4096 }));

    // Reference without any change.
    let mut plain = build(&spec);
    let mut unchanged = Vec::new();
    for block in input.chunks(1000) {
        plain.push("iq", InputData::C32(block)).unwrap();
        read_bb(&mut plain, &mut unchanged);
    }

    // With the timed change, in odd blocks that straddle the target.
    let mut p = build(&spec);
    p.set_param_at(
        "tune",
        "frequency_hz",
        json!(610_000.0),
        ParamTime::InputIndex {
            input: "iq".into(),
            index: at,
        },
    )
    .unwrap();
    let mut changed = Vec::new();
    for block in input.chunks(777) {
        p.push("iq", InputData::C32(block)).unwrap();
        read_bb(&mut p, &mut changed);
    }
    let expect = manual_split(decimation as usize, &input, at as usize, 610_000.0);
    assert_eq!(
        c32_bits(&changed),
        c32_bits(&expect),
        "equals the split by hand"
    );

    // The first output sample that differs from the unchanged run.
    let first_diff = changed
        .iter()
        .zip(&unchanged)
        .position(|(a, b)| c32_bits(&[*a]) != c32_bits(&[*b]))
        .unwrap() as u64;
    let mapped = at.div_ceil(decimation);
    assert!(
        first_diff.abs_diff(mapped) <= 1,
        "first changed output {first_diff}, mapped sample {mapped}"
    );
    if decimation == 1 {
        // The oscillator phase of sample `at` is still advanced with the old frequency; the new
        // one acts from the next sample on. Exactly one sample, no more.
        assert_eq!(first_diff, at + 1, "exact for decimation 1");
    }

    // The acknowledgement: on the output, at the mapped index, not late.
    let ev = events(&mut p);
    assert_eq!(ev.len(), 1, "{ev:?}");
    let node = p.node_id("tune").unwrap();
    assert_eq!(ev[0].output, p.output_id("bb").unwrap());
    assert_eq!(ev[0].index, mapped);
    let freq_index = p
        .node_id("tune")
        .map(|_| {
            sdrk_blocks::all_descriptions()
                .iter()
                .find(|d| d.kind == "ddc")
                .unwrap()
                .params
                .iter()
                .position(|q| q.name == "frequency_hz")
                .unwrap() as u16
        })
        .unwrap();
    assert_eq!(
        ev[0].kind,
        EventKind::ParamApplied {
            node,
            param: freq_index,
            value: ParamValue::F64(610_000.0),
            late: false
        }
    );
    assert_eq!(p.node_name(node), "tune");
}

#[test]
fn a_timed_change_is_exact_at_decimation_1() {
    timed_change(1);
}

#[test]
fn a_timed_change_is_exact_at_decimation_10() {
    timed_change(10);
}

#[test]
fn a_timed_change_in_the_past_is_applied_at_once_and_marked_late() {
    let mut p = build(&ddc_spec(
        1,
        json!({ "precision": "f32", "max_chunk": 4096 }),
    ));
    p.push("iq", InputData::C32(&tone(1000))).unwrap();
    p.set_param_at(
        "tune",
        "frequency_hz",
        json!(590_000.0),
        ParamTime::InputIndex {
            input: "iq".into(),
            index: 500,
        },
    )
    .unwrap();
    let ev = events(&mut p);
    assert_eq!(ev.len(), 1);
    let EventKind::ParamApplied { late, value, .. } = ev[0].kind else {
        panic!("{ev:?}")
    };
    assert!(late);
    assert_eq!(value, ParamValue::F64(590_000.0));
    assert_eq!(ev[0].index, 1000, "at the next sample");
    // Immediate is not late.
    p.set_param_at(
        "tune",
        "frequency_hz",
        json!(580_000.0),
        ParamTime::Immediate,
    )
    .unwrap();
    let ev = events(&mut p);
    let EventKind::ParamApplied { late, .. } = ev[0].kind else {
        panic!()
    };
    assert!(!late);
    // The config reads the new value back.
    assert_eq!(
        p.config().nodes["tune"].params["frequency_hz"],
        json!(580_000.0)
    );
}

#[test]
fn pending_changes_are_limited_and_validated() {
    let spec = ddc_spec(
        1,
        json!({ "precision": "f32", "max_chunk": 4096, "pending_params": 2 }),
    );
    let mut p = build(&spec);
    let at = |index| ParamTime::InputIndex {
        input: "iq".into(),
        index,
    };
    p.set_param_at("tune", "frequency_hz", json!(1.0), at(1000))
        .unwrap();
    p.set_param_at("tune", "frequency_hz", json!(2.0), at(2000))
        .unwrap();
    let err = p
        .set_param_at("tune", "frequency_hz", json!(3.0), at(3000))
        .unwrap_err();
    assert!(matches!(err, Error::PendingFull(_)), "{err}");
    // Changes apply in the order of their targets, whatever the order of the calls.
    let mut q = build(&ddc_spec(
        1,
        json!({ "precision": "f32", "max_chunk": 4096 }),
    ));
    q.set_param_at("tune", "frequency_hz", json!(2.0), at(2000))
        .unwrap();
    q.set_param_at("tune", "frequency_hz", json!(1.0), at(1000))
        .unwrap();
    q.push("iq", InputData::C32(&tone(3000))).unwrap();
    let ev = events(&mut q);
    let values: Vec<(u64, ParamValue)> = ev
        .iter()
        .map(|e| {
            let EventKind::ParamApplied { value, .. } = e.kind else {
                panic!()
            };
            (e.index, value)
        })
        .collect();
    assert_eq!(
        values,
        [(1000, ParamValue::F64(1.0)), (2000, ParamValue::F64(2.0))]
    );
    // Errors: unknown node, build parameter, wrong input, no anchor, bad value.
    assert!(matches!(
        q.set_param_at("nope", "frequency_hz", json!(1.0), ParamTime::Immediate),
        Err(Error::UnknownNode(_))
    ));
    assert!(matches!(
        q.set_param_at("tune", "decimation", json!(2), ParamTime::Immediate),
        Err(Error::Block { .. })
    ));
    assert!(matches!(
        q.set_param_at(
            "tune",
            "frequency_hz",
            json!(1.0),
            at(5000).clone_with_input("other")
        ),
        Err(Error::UnknownInput(_))
    ));
    assert!(matches!(
        q.set_param_at("tune", "frequency_hz", json!(1.0), ParamTime::UnixNs(5)),
        Err(Error::InvalidState(_))
    ));
    assert!(matches!(
        q.set_param_at("tune", "frequency_hz", json!("x"), ParamTime::Immediate),
        Err(Error::InvalidParameter { .. })
    ));
}

trait CloneWithInput {
    fn clone_with_input(&self, input: &str) -> ParamTime;
}

impl CloneWithInput for ParamTime {
    fn clone_with_input(&self, input: &str) -> ParamTime {
        match self {
            ParamTime::InputIndex { index, .. } => ParamTime::InputIndex {
                input: input.to_string(),
                index: *index,
            },
            other => other.clone(),
        }
    }
}

#[test]
fn a_change_at_an_absolute_time_uses_the_anchor() {
    let mut p = build(&ddc_spec(
        10,
        json!({ "precision": "f32", "max_chunk": 4096 }),
    ));
    let t0 = 5_000_000_000i64;
    let meta = BlockMeta {
        start: SampleIndex(0),
        anchor: Some(TimeAnchor {
            index: SampleIndex(0),
            unix_ns: t0,
            source: TimeSource::Host,
        }),
        ..BlockMeta::default()
    };
    // 2.4 MHz: 1 ms = 2400 samples. The change at t0 + 1.5 ms is input sample 3600.
    p.push_with_meta("iq", InputData::C32(&tone(1000)), &meta)
        .unwrap();
    let _ = events(&mut p);
    p.set_param_at(
        "tune",
        "frequency_hz",
        json!(610_000.0),
        ParamTime::UnixNs(t0 + 1_500_000),
    )
    .unwrap();
    p.push("iq", InputData::C32(&tone(4000))).unwrap();
    let ev: Vec<_> = events(&mut p)
        .into_iter()
        .filter(|e| matches!(e.kind, EventKind::ParamApplied { .. }))
        .collect();
    assert_eq!(ev.len(), 1);
    assert_eq!(ev[0].index, 360, "ceil(3600 / 10)");
}

#[test]
fn groups_and_settings_are_in_the_description_and_the_report() {
    // The settings of the queues round trip and are checked.
    let spec = PipelineSpec::from_json(&ddc_spec(
        1,
        json!({ "precision": "f32", "max_chunk": 64, "event_queue": 8, "pending_params": 3 }),
    ))
    .unwrap();
    assert_eq!(spec.settings.event_queue_len(), 8);
    assert_eq!(spec.settings.pending_params_len(), 3);
    let bad = ddc_spec(1, json!({ "event_queue": 0 }));
    let report = Registry::standard().validate_json(&bad, &BuildOptions::default());
    assert!(
        report
            .errors
            .iter()
            .any(|d| d.path == "settings.event_queue")
    );
}

#[test]
fn reset_restarts_the_indices_and_clears_events() {
    let mut p = build(&flow("fm_receiver"));
    let bytes = cu8_recording(8000);
    let meta = BlockMeta {
        start: SampleIndex(500),
        flags: BlockFlags::OVERFLOW,
        ..BlockMeta::default()
    };
    p.push_with_meta("rf", InputData::Raw(&bytes[..4000]), &meta)
        .unwrap();
    p.reset();
    assert_eq!(p.poll_event(), None);
    assert_eq!(p.input_index("rf").unwrap(), 0);
    assert_eq!(p.read_index("audio").unwrap(), 0);
    // After a reset the next block fixes the origin again.
    let meta = BlockMeta {
        start: SampleIndex(100),
        ..BlockMeta::default()
    };
    p.push_with_meta("rf", InputData::Raw(&bytes[..4000]), &meta)
        .unwrap();
    assert_eq!(p.poll_event(), None);
    assert_eq!(p.input_index("rf").unwrap(), 2100);
}

#[test]
fn the_mapping_rule_counts_the_samples_the_blocks_produce() {
    // After N input samples every node has produced ceil(N * L / M) samples: this is what the
    // mapping rule says about the first output that includes input sample N.
    for n in [1usize, 7, 10, 11, 12_345, 40_001] {
        let mut p = build(&flow("fm_receiver"));
        let bytes = cu8_recording(n);
        for chunk in bytes.chunks(2 * 4096) {
            p.push("rf", InputData::Raw(chunk)).unwrap();
            let mut sink = vec![0f32; 4096];
            while p.read_stream("audio", &mut sink).unwrap() > 0 {}
        }
        let stat = |name: &str| {
            p.stats()
                .iter()
                .find(|s| s.name == name)
                .unwrap()
                .samples_out as usize
        };
        assert_eq!(stat("rf__convert"), n, "convert, n = {n}");
        assert_eq!(stat("tune"), n.div_ceil(10), "ddc, n = {n}");
        assert_eq!(stat("demod"), n.div_ceil(10).div_ceil(5), "demod, n = {n}");
    }
}

#[test]
fn a_timed_change_is_acknowledged_on_every_output_reachable_from_the_node() {
    let at = 12_345u64;
    let spec = json!({
        "sdrk": 1, "name": "two-outputs",
        "settings": { "precision": "f32", "max_chunk": 4096 },
        "inputs": { "iq": { "type": "c32", "rate_hz": 2_400_000 } },
        "nodes": {
            "tune": { "kind": "ddc", "params": {
                "decimation": 10, "frequency_hz": 600_000.0, "cutoff_hz": 100_000.0 } },
            "demod": { "kind": "fm_demod", "params": {
                "audio_decimation": 5, "deemphasis_us": 50, "deviation_hz": 75_000 } }
        },
        "edges": [
            { "from": "iq", "to": "tune" },
            { "from": "tune", "to": "bb" },
            { "from": "tune", "to": "demod" },
            { "from": "demod", "to": "audio" }
        ],
        "outputs": {
            "bb": { "type": "stream", "capacity": 20000 },
            "audio": { "type": "stream", "capacity": 20000 }
        }
    })
    .to_string();
    let mut p = build(&spec);
    p.set_param_at(
        "tune",
        "frequency_hz",
        json!(610_000.0),
        ParamTime::InputIndex {
            input: "iq".into(),
            index: at,
        },
    )
    .unwrap();
    for block in tone(40_000).chunks(777) {
        p.push("iq", InputData::C32(block)).unwrap();
    }
    let ev: Vec<Event> = events(&mut p)
        .into_iter()
        .filter(|e| matches!(e.kind, EventKind::ParamApplied { .. }))
        .collect();
    assert_eq!(ev.len(), 2, "{ev:?}");
    let bb = ev
        .iter()
        .find(|e| e.output == p.output_id("bb").unwrap())
        .expect("ack on bb");
    let audio = ev
        .iter()
        .find(|e| e.output == p.output_id("audio").unwrap())
        .expect("ack on audio");
    assert_eq!(bb.index, at.div_ceil(10));
    assert_eq!(audio.index, at.div_ceil(50));
    assert_ne!(bb.index, audio.index);
}
