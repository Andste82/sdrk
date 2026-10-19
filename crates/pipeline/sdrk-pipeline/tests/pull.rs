//! Pull mode: the transmit path of `fm_transmitter.json`.
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

use sdrk_device::{BlockFlags, BlockMeta, SampleIndex};
use sdrk_pipeline::{
    BuildOptions, Error, EventKind, InputData, Pipeline, PullError, PullSource, Registry,
};
use serde_json::{Value, json};

use common::{build, flow};

/// `n` samples of a 1 kHz tone at 48 kHz, amplitude 0.8.
fn audio(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.8 * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 48_000.0).sin()) as f32)
        .collect()
}

/// Hands out a vector in pieces and remembers what was asked.
struct Source {
    data: Vec<f32>,
    pos: usize,
    asked: Vec<usize>,
}

impl Source {
    fn new(data: Vec<f32>) -> Self {
        Self {
            data,
            pos: 0,
            asked: Vec::new(),
        }
    }
}

impl PullSource for Source {
    fn provide(&mut self, input: &str, max_samples: usize) -> Result<InputData<'_>, PullError> {
        assert_eq!(input, "audio");
        assert!(max_samples > 0 && max_samples <= 4800, "{max_samples}");
        self.asked.push(max_samples);
        let n = max_samples.min(self.data.len() - self.pos);
        let from = self.pos;
        self.pos += n;
        Ok(InputData::F32(&self.data[from..from + n]))
    }
}

fn transmitter() -> Pipeline {
    build(&flow("fm_transmitter"))
}

fn transmitter_with(output: Value) -> Pipeline {
    let mut v: Value = serde_json::from_str(&flow("fm_transmitter")).unwrap();
    v["outputs"]["rf"] = output;
    build(&v.to_string())
}

/// The push-mode run: all `data` pushed in chunks of 1000, all bytes read.
fn push_run(data: &[f32]) -> Vec<u8> {
    let mut p = transmitter();
    let mut out = Vec::new();
    let mut buf = vec![0u8; 8192];
    for chunk in data.chunks(1000) {
        p.push("audio", InputData::F32(chunk)).unwrap();
        loop {
            let n = p.read_raw("rf", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
        }
    }
    out
}

/// A small deterministic generator for pull sizes.
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

#[test]
fn pulls_of_random_sizes_equal_the_push_mode_run() {
    let data = audio(6000);
    let reference = push_run(&data);
    assert!(reference.len() > 400_000);
    for seed in 1..=4u64 {
        let mut p = transmitter();
        let mut source = Source::new(data.clone());
        let mut rng = Lcg(seed);
        let mut got = Vec::new();
        let mut next_index = 0u64;
        while got.len() < 300_000 {
            let size = 2 * rng.next(20_000); // whole samples: cs8 has two bytes
            let mut buf = vec![0u8; size];
            let (n, meta) = p.pull("rf", &mut buf, &mut source).unwrap();
            assert_eq!(n, size, "the source is long enough");
            assert!(meta.flags.contains(BlockFlags::empty()));
            assert_eq!(meta.start.0, next_index, "the index continues");
            next_index += (n / 2) as u64;
            got.extend_from_slice(&buf[..n]);
        }
        assert_eq!(&got[..], &reference[..got.len()], "seed {seed}");
        assert!(source.asked.iter().all(|&n| n <= 4800));
    }
}

#[test]
fn an_odd_request_gets_whole_samples_only() {
    let mut p = transmitter();
    let mut source = Source::new(audio(2000));
    let mut buf = vec![0u8; 1001];
    let (n, _) = p.pull("rf", &mut buf, &mut source).unwrap();
    assert_eq!(n, 1000);
}

#[test]
fn a_pull_larger_than_the_output_queue_works() {
    // The queue holds 19 200 samples = 38 400 bytes by default.
    let data = audio(4000);
    let reference = push_run(&data);
    let mut p = transmitter();
    let mut source = Source::new(data);
    let mut buf = vec![0u8; 150_000];
    let (n, meta) = p.pull("rf", &mut buf, &mut source).unwrap();
    assert_eq!(n, 150_000);
    assert_eq!(meta.start, SampleIndex(0));
    assert_eq!(&buf[..], &reference[..150_000]);
}

#[test]
fn an_exhausted_source_is_filled_with_zeros_and_reported() {
    let mut p = transmitter();
    let mut source = Source::new(audio(200));
    let mut buf = vec![0x55u8; 40_000];
    let (n, meta) = p.pull("rf", &mut buf, &mut source).unwrap();
    assert_eq!(n, 40_000, "zeros policy: the request is filled");
    assert!(meta.flags.contains(BlockFlags::UNDERRUN));
    let ev: Vec<_> = std::iter::from_fn(|| p.poll_event()).collect();
    let under: Vec<_> = ev
        .iter()
        .filter(|e| e.kind == EventKind::Underrun)
        .collect();
    assert_eq!(under.len(), 1, "{ev:?}");
    assert_eq!(p.output_name(under[0].output), "rf");
    // The event is at the first silent sample: the end of what the graph produced. 200 samples
    // of audio lead to 10 000 samples of output, less what the nodes hold back.
    let first_silent = under[0].index as usize;
    assert!((9_000..=10_000).contains(&first_silent), "{first_silent}");
    assert!(buf[..2 * first_silent].iter().any(|b| *b != 0));
    assert!(
        buf[2 * first_silent..].iter().all(|b| *b == 0),
        "silence at the end"
    );
}

#[test]
fn the_short_policy_returns_fewer_samples() {
    let mut p = transmitter_with(json!({ "type": "stream", "format": "cs8", "underrun": "short" }));
    let mut source = Source::new(audio(200));
    let mut buf = vec![0u8; 40_000];
    let (n, meta) = p.pull("rf", &mut buf, &mut source).unwrap();
    assert!(n < 40_000 && n > 15_000, "{n}");
    assert!(!meta.flags.contains(BlockFlags::UNDERRUN));
    assert_eq!(p.poll_event(), None);
    // The next pull returns nothing more.
    let (n, _) = p.pull("rf", &mut buf, &mut source).unwrap();
    assert_eq!(n, 0);
}

#[test]
fn silence_is_a_zero_sample_in_the_format_of_the_output() {
    // Unsigned 8-bit samples have their zero in the middle of the range.
    let mut p = transmitter_with(json!({ "type": "stream", "format": "cu8", "underrun": "zeros" }));
    let mut source = Source::new(Vec::new());
    let mut buf = vec![0u8; 100];
    let (n, meta) = p.pull("rf", &mut buf, &mut source).unwrap();
    assert_eq!(n, 100);
    assert!(meta.flags.contains(BlockFlags::UNDERRUN));
    assert!(buf.iter().all(|b| *b == 128), "{:?}", &buf[..8]);
}

#[test]
fn pull_samples_serves_typed_outputs() {
    let mut p = build(&flow("resample_audio"));
    struct Ramp(Vec<f32>, usize);
    impl PullSource for Ramp {
        fn provide(&mut self, _i: &str, max: usize) -> Result<InputData<'_>, PullError> {
            let n = max.min(self.0.len() - self.1);
            let from = self.1;
            self.1 += n;
            Ok(InputData::F32(&self.0[from..from + n]))
        }
    }
    let spec =
        Registry::standard().validate_json(&flow("resample_audio"), &BuildOptions::default());
    assert!(spec.is_ok());
    let out_name = "audio_out";
    let mut src = Ramp(audio(20_000), 0);
    let mut out = vec![0f32; 3000];
    let (n, meta) = p.pull_samples::<f32>(out_name, &mut out, &mut src).unwrap();
    // The default policy of a typed output is `short`.
    assert!(n > 0 && n <= 3000);
    assert_eq!(meta.start, SampleIndex(0));
    // The wrong type and the wrong function are errors.
    let mut c = vec![sdrk::C32::default(); 4];
    assert!(matches!(
        p.pull_samples::<sdrk::C32>(out_name, &mut c, &mut src),
        Err(Error::WrongType { .. })
    ));
    let mut bytes = [0u8; 4];
    assert!(matches!(
        p.pull(out_name, &mut bytes, &mut src),
        Err(Error::WrongType { .. })
    ));
    assert!(matches!(
        p.pull("nope", &mut bytes, &mut src),
        Err(Error::UnknownOutput(_))
    ));
}

#[test]
fn the_lead_time_is_the_sum_of_the_delays_in_output_samples() {
    let p = transmitter();
    let lead = p.lead_time("rf").unwrap();
    assert_eq!(p.latency("rf").unwrap().group_delay, lead);
    // The report gives the delays in the input samples of every node.
    let report = p.report();
    let rate_out = 2_400_000.0f64;
    let mut sum = 0.0;
    for n in &report.nodes {
        let rate_in = n.rate_in.as_f64();
        sum += n.delay_samples_in.as_f64() * rate_out / rate_in;
    }
    assert!(
        (lead.as_f64() - sum).abs() < 1e-6,
        "{} vs {sum}",
        lead.as_f64()
    );
    assert!(lead.as_f64() > 10.0);
    // The report of validate has the same figure.
    let validated =
        Registry::standard().validate_json(&flow("fm_transmitter"), &BuildOptions::default());
    let r = validated.outputs.iter().find(|o| o.name == "rf").unwrap();
    assert_eq!(r.group_delay.reduced(), lead.reduced());
    assert!(r.max_buffered < 100, "{}", r.max_buffered);
}

/// Pulls until a block carries `BURST_END`; returns the samples and the flags of each block.
fn pull_burst(
    p: &mut Pipeline,
    source: &mut Source,
    mut limit: usize,
) -> (Vec<u8>, Vec<BlockMeta>) {
    let mut bytes = Vec::new();
    let mut metas = Vec::new();
    while limit > 0 {
        let mut buf = vec![0u8; 5000];
        let (n, meta) = p.pull("rf", &mut buf, source).unwrap();
        if n == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..n]);
        metas.push(meta);
        limit -= 1;
        if meta.flags.contains(BlockFlags::BURST_END) {
            break;
        }
    }
    (bytes, metas)
}

#[test]
fn a_burst_has_a_start_an_end_a_tail_and_the_index_continues() {
    let n_audio = 1000usize;
    let data = audio(n_audio);
    let mut p = transmitter_with(json!({ "type": "stream", "format": "cs8", "underrun": "short" }));
    let lead = p.lead_time("rf").unwrap();
    let lead_ceil = lead.reduced().num().div_ceil(lead.reduced().den());

    // Between bursts nothing is produced.
    let mut source = Source::new(data.clone());
    let mut buf = vec![0u8; 1000];
    // (Continuous mode before the first burst would produce; the first call arms the burst.)
    p.begin_burst("rf", Some(1_700_000_000_000_000_000))
        .unwrap();
    // A second begin while one is active is an error.
    assert!(matches!(
        p.begin_burst("rf", None),
        Err(Error::InvalidState(_))
    ));

    // First, the data of the burst; the source is exhausted after n_audio samples.
    let mut first = Vec::new();
    let mut metas = Vec::new();
    loop {
        let (n, meta) = p.pull("rf", &mut buf, &mut source).unwrap();
        if n == 0 {
            break;
        }
        first.extend_from_slice(&buf[..n]);
        metas.push(meta);
    }
    assert_eq!(metas[0].start, SampleIndex(0));
    assert!(metas[0].flags.contains(BlockFlags::BURST_START));
    assert_eq!(metas[0].tx_time_ns, Some(1_700_000_000_000_000_000));
    assert!(metas[1..].iter().all(|m| m.flags == BlockFlags::empty()));
    // Every audio sample leads to 50 output samples, less what the nodes hold back.
    let data_samples = first.len() / 2;
    assert!(
        (n_audio * 50 - 1000..=n_audio * 50).contains(&data_samples),
        "{data_samples}"
    );

    // End of the burst: the tail comes out.
    p.end_burst("rf").unwrap();
    let (tail, tail_metas) = pull_burst(&mut p, &mut source, 100);
    let last = tail_metas.last().unwrap();
    assert!(last.flags.contains(BlockFlags::BURST_END));
    assert!(
        tail_metas[..tail_metas.len() - 1]
            .iter()
            .all(|m| !m.flags.contains(BlockFlags::BURST_END))
    );
    // The burst is n_audio * 50 samples long plus the tail of ceil(lead) samples (+-1).
    let total = data_samples + tail.len() / 2;
    let tail_len = total as i64 - (n_audio * 50) as i64;
    assert!(
        (tail_len - lead_ceil as i64).abs() <= 1,
        "tail {tail_len}, lead {lead_ceil}"
    );
    // After the end nothing is produced until the next begin.
    let (n, _) = p.pull("rf", &mut buf, &mut source).unwrap();
    assert_eq!(n, 0);
    let events: Vec<_> = std::iter::from_fn(|| p.poll_event()).collect();
    let starts: Vec<_> = events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::BurstStart { .. }))
        .collect();
    let ends: Vec<_> = events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::BurstEnd))
        .collect();
    assert_eq!((starts.len(), ends.len()), (1, 1));
    assert_eq!(starts[0].index, 0);
    assert_eq!(ends[0].index, total as u64 - 1);

    // A second burst with the same audio: the same bytes (the nodes were reset), the index
    // continues after the first burst.
    let mut source2 = Source::new(data);
    p.begin_burst("rf", None).unwrap();
    let mut second = Vec::new();
    let mut metas2 = Vec::new();
    loop {
        let (n, meta) = p.pull("rf", &mut buf, &mut source2).unwrap();
        if n == 0 {
            break;
        }
        second.extend_from_slice(&buf[..n]);
        metas2.push(meta);
    }
    p.end_burst("rf").unwrap();
    let (tail2, tail_metas2) = pull_burst(&mut p, &mut source2, 100);
    assert!(
        metas2[0].start.0 >= total as u64,
        "the index never goes back: {:?}",
        metas2[0]
    );
    assert!(
        metas2[0].start.0 <= total as u64 + 60,
        "and does not jump far"
    );
    assert!(metas2[0].flags.contains(BlockFlags::BURST_START));
    assert_eq!(metas2[0].tx_time_ns, None);
    assert!(
        tail_metas2
            .last()
            .unwrap()
            .flags
            .contains(BlockFlags::BURST_END)
    );
    assert_eq!(second, first, "burst data");
    assert_eq!(tail2, tail, "burst tail");
}

#[test]
fn a_block_never_spans_a_burst_boundary() {
    let mut p = transmitter_with(json!({ "type": "stream", "format": "cs8", "underrun": "short" }));
    let mut source = Source::new(audio(4000));
    // Continuous mode first: some samples are pulled, some stay buffered.
    let mut buf = vec![0u8; 3001 * 2];
    let (n, meta) = p.pull("rf", &mut buf, &mut source).unwrap();
    assert_eq!((n, meta.flags), (3001 * 2, BlockFlags::empty()));
    let buffered = p.available("rf").unwrap();
    p.begin_burst("rf", Some(7)).unwrap();
    let mut big = vec![0u8; 60_000];
    let (n1, m1) = p.pull("rf", &mut big, &mut source).unwrap();
    if buffered > 0 {
        // The samples that were there before `begin_burst` come first, without the flag.
        assert_eq!(n1, buffered);
        assert_eq!(m1.flags, BlockFlags::empty());
        let (n2, m2) = p.pull("rf", &mut big, &mut source).unwrap();
        assert!(n2 > 0);
        assert!(m2.flags.contains(BlockFlags::BURST_START));
        assert_eq!(m2.start.0, m1.start.0 + (n1 / 2) as u64);
    } else {
        assert!(m1.flags.contains(BlockFlags::BURST_START));
    }
}

#[test]
fn ending_a_burst_that_never_produced_cancels_it() {
    let mut p = transmitter();
    assert!(matches!(p.end_burst("rf"), Err(Error::InvalidState(_))));
    p.begin_burst("rf", None).unwrap();
    p.end_burst("rf").unwrap();
    let mut source = Source::new(audio(100));
    let mut buf = vec![0u8; 100];
    // Idle: nothing is produced, the source is not asked.
    let (n, _) = p.pull("rf", &mut buf, &mut source).unwrap();
    assert_eq!(n, 0);
    assert!(source.asked.is_empty());
    assert!(matches!(
        p.begin_burst("nope", None),
        Err(Error::UnknownOutput(_))
    ));
}

#[test]
fn the_device_status_becomes_events() {
    let mut p = transmitter();
    let meta = BlockMeta {
        start: SampleIndex(5000),
        flags: BlockFlags::UNDERRUN.union(BlockFlags::LATE),
        ..BlockMeta::default()
    };
    p.device_status("rf", &meta).unwrap();
    let ev: Vec<_> = std::iter::from_fn(|| p.poll_event()).collect();
    assert_eq!(ev.len(), 2);
    assert!(
        ev.iter()
            .all(|e| e.index == 5000 && p.output_name(e.output) == "rf")
    );
    assert!(ev.iter().any(|e| e.kind == EventKind::Underrun));
    assert!(ev.iter().any(|e| e.kind == EventKind::Late));
    assert!(matches!(
        p.device_status("nope", &meta),
        Err(Error::UnknownOutput(_))
    ));
}

#[test]
fn a_source_that_returns_too_much_is_an_error() {
    struct Greedy(Vec<f32>);
    impl PullSource for Greedy {
        fn provide(&mut self, _input: &str, _max: usize) -> Result<InputData<'_>, PullError> {
            Ok(InputData::F32(&self.0))
        }
    }
    let mut p = transmitter();
    let mut src = Greedy(vec![0.0; 4800]);
    let mut buf = vec![0u8; 10];
    assert!(matches!(
        p.pull("rf", &mut buf, &mut src),
        Err(Error::ChunkTooLarge { .. })
    ));
    struct Failing;
    impl PullSource for Failing {
        fn provide(&mut self, _input: &str, _max: usize) -> Result<InputData<'_>, PullError> {
            Err(Error::InvalidState("source failed".into()))
        }
    }
    assert!(matches!(
        p.pull("rf", &mut buf, &mut Failing),
        Err(Error::InvalidState(_))
    ));
}
