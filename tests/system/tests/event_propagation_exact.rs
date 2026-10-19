//! Events (gap, anchor, retune, overflow) at 50 random places of a recording pushed through
//! `fm_receiver.json`: each one appears on `audio` at exactly the index that the mapping rule of
//! `docs/timing.md` gives, node by node.
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

use sdrk_device::{BlockFlags, BlockMeta, SampleIndex, TimeAnchor, TimeSource, formats};
use sdrk_pipeline::{EventKind, InputData, NodeStats};
use sdrk_system_tests::{build, encode_bytes, flow, fm_station};
use sdrk_testkit::Pcg32;

const SAMPLES: usize = 1_200_000;
const MAX_CHUNK: usize = 4096;
const EVENTS: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Gap(u64),
    Anchor,
    Retune,
    Overflow,
}

/// One node of the audio path as the mapping rule sees it.
struct Hop {
    /// Name in the pipeline statistics.
    name: &'static str,
    /// Ratio out/in as `(L, M)`.
    ratio: (u64, u64),
    n0_in: u64,
    n0_out: u64,
    /// Samples the node had produced (statistics) when the current origin was set.
    produced_at_origin: u64,
}

impl Hop {
    fn new(name: &'static str, ratio: (u64, u64)) -> Self {
        Self {
            name,
            ratio,
            n0_in: 0,
            n0_out: 0,
            produced_at_origin: 0,
        }
    }

    /// `n0_out + ceil((n - n0_in) L / M)`.
    fn map(&self, n: u64) -> u64 {
        self.n0_out + ((n - self.n0_in) * self.ratio.0).div_ceil(self.ratio.1)
    }
}

fn produced(stats: &[NodeStats], name: &str) -> u64 {
    stats.iter().find(|s| s.name == name).unwrap().samples_out
}

#[test]
fn event_propagation_exact() {
    let mut spec = flow("fm_receiver");
    spec.settings.max_chunk = MAX_CHUNK;
    let mut p = build(&spec);

    // The recording and the plan: block sizes and which blocks carry an event.
    let iq = fm_station(600_000.0, 1000.0, 0.9, 0, SAMPLES);
    let bytes = encode_bytes(&formats::CU8, &iq);
    let mut rng = Pcg32::new(2024, 7);
    let mut sizes = Vec::new();
    let mut total = 0usize;
    while total < SAMPLES {
        let n = (1000 + rng.below(3097) as usize).min(SAMPLES - total);
        sizes.push(n);
        total += n;
    }
    let mut chosen = std::collections::BTreeSet::new();
    while chosen.len() < EVENTS {
        chosen.insert(1 + rng.below(sizes.len() as u32 - 1) as usize);
    }
    let plan: Vec<Option<Kind>> = (0..sizes.len())
        .map(|k| {
            chosen.contains(&k).then(|| match rng.below(4) {
                0 => Kind::Gap(1 + u64::from(rng.below(30_000))),
                1 => Kind::Anchor,
                2 => Kind::Retune,
                _ => Kind::Overflow,
            })
        })
        .collect();
    assert_eq!(plan.iter().flatten().count(), EVENTS);

    // Model of the audio path: convert (1/1), ddc (1/10), fm_demod (1/5).
    let mut hops = [
        Hop::new("rf__convert", (1, 1)),
        Hop::new("tune", (1, 10)),
        Hop::new("demod", (1, 5)),
    ];
    let mut expected: Vec<(u64, Kind)> = Vec::new();
    let mut expected_spectrum: Vec<(u64, Kind)> = Vec::new();

    let mut actual = Vec::new();
    let mut audio = vec![0f32; 4096];
    let mut frame = vec![0f32; 1024];
    let mut audio_total = 0usize;
    let mut position = 0usize; // sample position in the recording
    let mut index = 0u64; // sample index of the stream
    for (k, &n) in sizes.iter().enumerate() {
        let mut meta = BlockMeta {
            start: SampleIndex(index),
            ..BlockMeta::default()
        };
        let kind = plan[k];
        match kind {
            Some(Kind::Gap(missing)) => {
                // origins move: out_end + ceil(missing L / M), node by node
                let stats = p.stats().to_vec();
                let mut gap = missing;
                let new_start = index + missing;
                let mut next_in = new_start;
                for h in &mut hops {
                    let out_end = h.n0_out + produced(&stats, h.name) - h.produced_at_origin;
                    gap = (gap * h.ratio.0).div_ceil(h.ratio.1);
                    h.n0_in = next_in;
                    h.n0_out = out_end + gap;
                    h.produced_at_origin = produced(&stats, h.name);
                    next_in = h.n0_out;
                }
                index = new_start;
                meta.start = SampleIndex(index);
                meta.flags = BlockFlags::GAP_BEFORE;
            }
            Some(Kind::Anchor) => {
                meta.anchor = Some(TimeAnchor {
                    index: SampleIndex(index),
                    unix_ns: 1_700_000_000_000_000_000 + (index as i64) * 417,
                    source: TimeSource::Host,
                });
            }
            Some(Kind::Retune) => meta.flags = BlockFlags::RETUNED,
            Some(Kind::Overflow) => meta.flags = BlockFlags::OVERFLOW,
            None => {}
        }
        if let Some(kind) = kind {
            let audio_index = hops.iter().fold(index, |i, h| h.map(i));
            expected.push((audio_index, kind));
            // The spectrum is a frames output: its unit is the sample at the input of the FFT
            // node, which is the output of the convert node.
            expected_spectrum.push((hops[0].map(index), kind));
        }
        let from = position * 2;
        p.push_with_meta("rf", InputData::Raw(&bytes[from..from + 2 * n]), &meta)
            .unwrap();
        position += n;
        index += n as u64;
        while let Some(e) = p.poll_event() {
            actual.push((p.output_name(e.output).to_string(), e));
        }
        loop {
            let got = p.read_stream::<f32>("audio", &mut audio).unwrap();
            if got == 0 {
                break;
            }
            audio_total += got;
        }
        while p.read_frame("spectrum", &mut frame).unwrap().is_some() {}
    }
    assert!(audio_total > 20_000);

    let to_kind = |kind: EventKind| match kind {
        EventKind::Gap { missing } => Kind::Gap(missing),
        EventKind::Anchor(_) => Kind::Anchor,
        EventKind::Retuned => Kind::Retune,
        EventKind::Overflow => Kind::Overflow,
        other => panic!("unexpected event {other:?}"),
    };
    let on = |name: &str| -> Vec<(u64, Kind)> {
        actual
            .iter()
            .filter(|(o, _)| o == name)
            .map(|(_, e)| (e.index, to_kind(e.kind)))
            .collect()
    };
    let audio_events = on("audio");
    assert_eq!(
        audio_events.len(),
        EVENTS,
        "one event per injected event on `audio`"
    );
    for (k, (got, want)) in audio_events.iter().zip(&expected).enumerate() {
        assert_eq!(got, want, "event {k} on `audio`");
    }
    assert_eq!(on("spectrum"), expected_spectrum, "events on `spectrum`");
    assert_eq!(actual.len(), 2 * EVENTS, "no other events");
    let gaps = expected
        .iter()
        .filter(|(_, k)| matches!(k, Kind::Gap(_)))
        .count();
    assert!(gaps >= 5, "{gaps} gaps in the plan");
}
