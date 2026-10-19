//! The two-thread architecture probe: a pipeline split by groups and run on two threads,
//! connected by wire-encoded messages, gives the result of the single-threaded pipeline.
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

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, Sender, channel};

use sdrk::C32;
use sdrk_blocks::FrameMeta;
use sdrk_device::{BlockFlags, BlockMeta, SampleIndex, TimeAnchor, TimeSource};
use sdrk_pipeline::wire::{Hello, Message, MessageType, decode_stream, encode};
use sdrk_pipeline::{
    BuildOptions, Event, EventKind, InputData, Pipeline, PipelineSpec, Registry, split_by_groups,
};

use common::{bits, build, cu8_recording, flow};

const RF: u16 = 1;
const EDGE: u16 = 2;
const EDGE_NAME: &str = "edge__tune__demod";
const T0: i64 = 1_700_000_000_000_000_000;

/// A deterministic generator.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, max: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        1 + ((self.0 >> 33) as usize) % max
    }
}

/// The metadata of block number `k` starting at sample `start`.
fn meta_for(k: usize, start: u64) -> BlockMeta {
    let mut flags = BlockFlags::empty();
    if k % 5 == 3 {
        flags.insert(BlockFlags::OVERFLOW);
    }
    if k % 7 == 4 {
        flags.insert(BlockFlags::RETUNED);
    }
    BlockMeta {
        start: SampleIndex(start),
        flags,
        anchor: (k == 0).then_some(TimeAnchor {
            index: SampleIndex(0),
            unix_ns: T0,
            source: TimeSource::Host,
        }),
        tx_time_ns: None,
    }
}

/// What a run produces: audio, spectrum frames, and the events by output name.
#[derive(Debug, Default)]
struct Collected {
    audio: Vec<f32>,
    frames: Vec<(Vec<f32>, FrameMeta)>,
    events: Vec<(String, u64, EventKind)>,
}

fn drain(p: &mut Pipeline, c: &mut Collected) {
    let mut buf = vec![0f32; 4096];
    let mut frame = vec![0f32; 1024];
    loop {
        let n = p.read_stream("audio", &mut buf).unwrap();
        if n == 0 {
            break;
        }
        c.audio.extend_from_slice(&buf[..n]);
    }
    while let Some(m) = p.read_frame("spectrum", &mut frame).unwrap() {
        c.frames.push((frame.clone(), m));
    }
    while let Some(e) = p.poll_event() {
        c.events
            .push((p.output_name(e.output).to_string(), e.index, e.kind));
    }
}

/// The reference: one pipeline, one thread.
fn single(plan: &[usize], bytes: &[u8]) -> Collected {
    let mut p = build(&flow("fm_receiver_grouped"));
    let mut c = Collected::default();
    let (mut pos, mut index) = (0usize, 0u64);
    for (k, &chunk) in plan.iter().enumerate() {
        let n = chunk.min(bytes.len() / 2 - pos);
        if n == 0 {
            break;
        }
        let meta = meta_for(k, index);
        p.push_with_meta("rf", InputData::Raw(&bytes[2 * pos..2 * (pos + n)]), &meta)
            .unwrap();
        pos += n;
        index += n as u64;
        drain(&mut p, &mut c);
    }
    c
}

/// Cuts a byte stream at random places and sends the pieces.
struct Sender2 {
    tx: Sender<Vec<u8>>,
    pending: Vec<u8>,
    rng: Lcg,
}

impl Sender2 {
    fn new(tx: Sender<Vec<u8>>, seed: u64) -> Self {
        Self {
            tx,
            pending: Vec::new(),
            rng: Lcg(seed),
        }
    }

    fn send(&mut self, m: &Message) {
        encode(m, &mut self.pending);
        // Flush a random-sized piece now and then, in the middle of a message if it happens.
        if self.rng.next(3) == 1 {
            let cut = self
                .rng
                .next(self.pending.len().max(1))
                .min(self.pending.len());
            let rest = self.pending.split_off(cut);
            let piece = std::mem::replace(&mut self.pending, rest);
            self.tx.send(piece).unwrap();
        }
    }

    fn finish(self) {
        if !self.pending.is_empty() {
            self.tx.send(self.pending).unwrap();
        }
    }
}

/// Receives messages from a channel until it closes.
fn receive(rx: &Receiver<Vec<u8>>, mut each: impl FnMut(Message)) {
    let mut buf: Vec<u8> = Vec::new();
    while let Ok(piece) = rx.recv() {
        buf.extend_from_slice(&piece);
        let (messages, used) = decode_stream(&buf).unwrap();
        buf.drain(..used);
        for m in messages {
            each(m);
        }
    }
    assert!(buf.is_empty(), "the stream ends between messages");
}

/// Turns the events that came before a block into the metadata of the block.
#[derive(Default)]
struct MetaBuilder {
    flags: BlockFlags,
    anchor: Option<TimeAnchor>,
}

impl MetaBuilder {
    fn event(&mut self, e: &Event) {
        match e.kind {
            EventKind::Overflow => self.flags.insert(BlockFlags::OVERFLOW),
            EventKind::Retuned => self.flags.insert(BlockFlags::RETUNED),
            EventKind::Anchor(a) => self.anchor = Some(a),
            other => panic!("the probe sends no {other:?}"),
        }
    }

    fn take(&mut self, start: u64) -> BlockMeta {
        let m = BlockMeta {
            start: SampleIndex(start),
            flags: self.flags,
            anchor: self.anchor.take(),
            tx_time_ns: None,
        };
        self.flags = BlockFlags::empty();
        m
    }
}

/// Group `front`: the tuner. Reads the rf stream, writes the edge stream.
fn front(spec: &PipelineSpec, rx: &Receiver<Vec<u8>>, tx: Sender<Vec<u8>>, seed: u64) {
    let mut p = Registry::standard()
        .build(spec, &BuildOptions::default())
        .unwrap();
    let out_id = p.output_id(EDGE_NAME).unwrap();
    let mut out = Sender2::new(tx, seed);
    // The description of the stream the receiver gets.
    let edge_rate = spec.outputs.get(EDGE_NAME).map(|_| ());
    assert!(edge_rate.is_some());
    out.send(&Message::hello(&Hello::samples(
        EDGE, EDGE_NAME, "c32", 240_000, 1,
    )));
    let mut meta = MetaBuilder::default();
    let mut pending: VecDeque<Event> = VecDeque::new();
    let mut edge = vec![C32::default(); 4096];
    receive(rx, |m| match m.kind {
        MessageType::Hello => assert_eq!(m.to_hello().unwrap().name, "rf"),
        MessageType::Event => meta.event(&m.to_event(out_id).unwrap()),
        MessageType::Samples => {
            let data = m.to_data().unwrap();
            let block = meta.take(m.start);
            // `m.flags` carries the flags of the block; the events carried the anchor.
            let block = BlockMeta {
                flags: m.flags.union(block.flags),
                ..block
            };
            p.push_with_meta("rf", data.as_input(), &block).unwrap();
            while let Some(e) = p.poll_event() {
                assert_eq!(e.output, out_id);
                pending.push_back(e);
            }
            // Everything available is sent, split at the events so that every event is at
            // the start of a block.
            loop {
                let first = p.read_index(EDGE_NAME).unwrap();
                let n = p.read_stream(EDGE_NAME, &mut edge).unwrap();
                if n == 0 {
                    break;
                }
                let mut at = 0usize;
                while at < n {
                    let here = first + at as u64;
                    match pending.front() {
                        Some(e) if e.index <= here => {
                            let e = pending.pop_front().unwrap();
                            out.send(&Message::event(EDGE, &e));
                        }
                        Some(e) if e.index < first + n as u64 => {
                            let upto = (e.index - first) as usize;
                            out.send(&Message::samples_c32(
                                EDGE,
                                here,
                                BlockFlags::empty(),
                                &edge[at..upto],
                            ));
                            at = upto;
                        }
                        _ => {
                            out.send(&Message::samples_c32(
                                EDGE,
                                here,
                                BlockFlags::empty(),
                                &edge[at..n],
                            ));
                            at = n;
                        }
                    }
                }
            }
        }
        other => panic!("unexpected {other:?}"),
    });
    assert!(pending.is_empty(), "no event beyond the data: {pending:?}");
    out.finish();
}

/// Group `audio`: demodulation and spectrum. Reads rf and the edge stream.
fn audio_group(
    spec: &PipelineSpec,
    rx_rf: &Receiver<Vec<u8>>,
    rx_edge: &Receiver<Vec<u8>>,
) -> Collected {
    let mut p = Registry::standard()
        .build(spec, &BuildOptions::default())
        .unwrap();
    let any = p.output_id("audio").unwrap();
    let mut c = Collected::default();
    let mut meta = MetaBuilder::default();
    receive(rx_rf, |m| match m.kind {
        MessageType::Hello => {}
        MessageType::Event => meta.event(&m.to_event(any).unwrap()),
        MessageType::Samples => {
            let data = m.to_data().unwrap();
            let block = meta.take(m.start);
            let block = BlockMeta {
                flags: m.flags.union(block.flags),
                ..block
            };
            p.push_with_meta("rf", data.as_input(), &block).unwrap();
            drain(&mut p, &mut c);
        }
        other => panic!("unexpected {other:?}"),
    });
    let mut meta = MetaBuilder::default();
    receive(rx_edge, |m| match m.kind {
        MessageType::Hello => {
            let h = m.to_hello().unwrap();
            assert_eq!(
                (h.name.as_str(), h.sample_type.as_deref()),
                (EDGE_NAME, Some("c32"))
            );
            // The description matches the input of this group.
            let input = &spec.inputs[EDGE_NAME];
            assert_eq!(
                h.rate_hz,
                format!("{}/{}", input.rate_hz.0.num(), input.rate_hz.0.den())
            );
        }
        MessageType::Event => meta.event(&m.to_event(any).unwrap()),
        MessageType::Samples => {
            let data = m.to_data().unwrap();
            let block = meta.take(m.start);
            p.push_with_meta(EDGE_NAME, data.as_input(), &block)
                .unwrap();
            drain(&mut p, &mut c);
        }
        other => panic!("unexpected {other:?}"),
    });
    drain(&mut p, &mut c);
    c
}

fn two_threads(plan: &[usize], bytes: &[u8], seed: u64) -> Collected {
    let registry = Registry::standard();
    let spec = PipelineSpec::from_json(&flow("fm_receiver_grouped")).unwrap();
    let parts = split_by_groups(&spec, &registry).unwrap();
    let audio_spec = &parts.iter().find(|(g, _)| g == "audio").unwrap().1;
    let front_spec = &parts.iter().find(|(g, _)| g == "front").unwrap().1;

    let (tx_front, rx_front) = channel::<Vec<u8>>();
    let (tx_rf, rx_rf) = channel::<Vec<u8>>();
    let (tx_edge, rx_edge) = channel::<Vec<u8>>();

    // The id of an output is only needed to build an event message; it is not sent.
    let reference = build(&flow("fm_receiver_grouped"));
    let any = reference.output_id("audio").unwrap();

    std::thread::scope(|s| {
        let a = s.spawn(move || front(front_spec, &rx_front, tx_edge, seed + 1));
        let b = s.spawn(move || audio_group(audio_spec, &rx_rf, &rx_edge));
        // The source: the same blocks to both groups that read the input.
        let mut to_front = Sender2::new(tx_front, seed + 2);
        let mut to_audio = Sender2::new(tx_rf, seed + 3);
        let hello = Message::hello(&Hello::raw(RF, "rf", "cu8", 2_400_000, 1));
        to_front.send(&hello);
        to_audio.send(&hello);
        let (mut pos, mut index) = (0usize, 0u64);
        for (k, &chunk) in plan.iter().enumerate() {
            let n = chunk.min(bytes.len() / 2 - pos);
            if n == 0 {
                break;
            }
            let meta = meta_for(k, index);
            let block = &bytes[2 * pos..2 * (pos + n)];
            for out in [&mut to_front, &mut to_audio] {
                if let Some(a) = meta.anchor {
                    out.send(&Message::event(
                        RF,
                        &Event {
                            output: any,
                            index: a.index.0,
                            kind: EventKind::Anchor(a),
                        },
                    ));
                }
                out.send(&Message::samples_raw(RF, index, meta.flags, n, block));
            }
            pos += n;
            index += n as u64;
        }
        to_front.finish();
        to_audio.finish();
        a.join().unwrap();
        b.join().unwrap()
    })
}

/// Random chunk sizes in samples, 1 to 4096, that add up to more than `total`.
fn random_plan(seed: u64, total: usize) -> Vec<usize> {
    let mut rng = Lcg(seed);
    let mut plan = Vec::new();
    let mut sum = 0;
    while sum < total {
        let n = rng.next(4096);
        plan.push(n);
        sum += n;
    }
    plan
}

#[test]
fn the_split_pipeline_on_two_threads_equals_the_single_pipeline() {
    let total = 160_000;
    let bytes = cu8_recording(total);
    for seed in 1..=10u64 {
        let plan = random_plan(seed * 7919, total);
        let one = single(&plan, &bytes);
        let two = two_threads(&plan, &bytes, seed * 31);
        assert!(one.audio.len() > 3000, "the run produced audio");
        assert!(!one.frames.is_empty());
        assert_eq!(bits(&one.audio), bits(&two.audio), "plan {seed}: audio");
        assert_eq!(
            one.frames.len(),
            two.frames.len(),
            "plan {seed}: frame count"
        );
        for (a, b) in one.frames.iter().zip(&two.frames) {
            assert_eq!(bits(&a.0), bits(&b.0), "plan {seed}: spectrum");
            assert_eq!(a.1, b.1, "plan {seed}: frame metadata");
        }
        // The events per output, in order.
        for output in ["audio", "spectrum"] {
            let pick = |c: &Collected| -> Vec<(u64, EventKind)> {
                c.events
                    .iter()
                    .filter(|(o, _, _)| o == output)
                    .map(|(_, i, k)| (*i, *k))
                    .collect()
            };
            let (a, b) = (pick(&one), pick(&two));
            assert!(a.len() >= 3, "plan {seed}: {output} has events: {a:?}");
            assert_eq!(a, b, "plan {seed}: events of {output}");
        }
        assert!(
            one.events
                .iter()
                .any(|(o, _, k)| o == "audio" && matches!(k, EventKind::Anchor(_)))
        );
    }
}
