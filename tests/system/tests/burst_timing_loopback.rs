//! Burst timing through the loopback channel: two bursts of a constant carrier are written with
//! transmit times; the receiver flow (down-converter output) shows them at the delayed time.
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

use num_complex::Complex64;
use sdrk::C32;
use sdrk_device::RxStream;
use sdrk_device::{BlockFlags, BlockMeta, Rate, TxStream, formats};
use sdrk_pipeline::{EdgeSpec, FormatName, InputData, OutputSpec, OutputType};
use sdrk_system_tests::{IF_RATE, build, drain_events, encode_bytes, flow, set_input_format};
use sdrk_testkit::channel::{Loopback, LoopbackParams};

const T0_NS: i64 = 1_700_000_000_000_000_000;
const DELAY_SAMPLES: f64 = 1234.37;
const BURST_SECONDS: f64 = 0.1;
const CARRIER_HZ: f64 = 600_000.0;
const BURST_STARTS_NS: [i64; 2] = [1_000_000_000, 1_500_000_000];
/// Output sample period of the down-converter (decimation 10 at 2.4 MS/s) in nanoseconds.
const OUTPUT_PERIOD_NS: f64 = 1e9 / 240_000.0;

#[test]
fn burst_timing_loopback() {
    // Channel
    let mut params = LoopbackParams::new(Rate::hz(IF_RATE).unwrap(), formats::CS8);
    params.t0_ns = T0_NS;
    params.delay_samples = DELAY_SAMPLES;
    params.noise_sigma = 0.002;
    params.seed = 99;
    let (mut tx, mut rx) = Loopback::new(params).unwrap().split();

    // Transmitter: bursts of a constant carrier at +600 kHz, written with their start times.
    let n = (IF_RATE as f64 * BURST_SECONDS) as usize;
    let carrier: Vec<Complex64> = (0..n)
        .map(|i| {
            let t = i as f64 * CARRIER_HZ / IF_RATE as f64;
            let ph = 2.0 * std::f64::consts::PI * (t - t.floor());
            Complex64::new(0.8 * ph.cos(), 0.8 * ph.sin())
        })
        .collect();
    let bytes = encode_bytes(&formats::CS8, &carrier);
    for start in BURST_STARTS_NS {
        let first = BlockMeta {
            flags: BlockFlags::BURST_START,
            tx_time_ns: Some(T0_NS + start),
            ..BlockMeta::default()
        };
        let half = bytes.len() / 2;
        tx.write(&bytes[..half], &first).unwrap();
        let last = BlockMeta {
            flags: BlockFlags::BURST_END,
            ..BlockMeta::default()
        };
        tx.write(&bytes[half..], &last).unwrap();
    }
    assert_eq!(tx.poll_status().unwrap(), None);

    // Receiver: the receiver flow with the output of the down-converter as a stream.
    let mut spec = flow("fm_receiver");
    set_input_format(&mut spec, "rf", FormatName::Cs8);
    spec.edges.push(EdgeSpec {
        from: "tune".into(),
        to: "iq".into(),
    });
    spec.outputs.insert(
        "iq".into(),
        OutputSpec {
            output_type: OutputType::Stream,
            format: None,
            capacity: Some(1 << 16),
            queue: None,
            drop: None,
            underrun: None,
            description: None,
        },
    );
    let mut p = build(&spec);
    assert_eq!(p.output_rate("iq").unwrap(), Rate::hz(240_000).unwrap());

    let total_seconds = 1.7;
    let total = (IF_RATE as f64 * total_seconds) as usize;
    let mut power: Vec<f64> = Vec::with_capacity(total / 10 + 1000);
    let mut buf = vec![0u8; 4096 * 2];
    let mut iq = vec![C32::default(); 4096];
    let mut audio = vec![0f32; 4096];
    let mut frame = vec![0f32; 1024];
    let mut events = Vec::new();
    let mut got = 0usize;
    while got < total {
        let want = (total - got).min(4096) * 2;
        let (k, meta) = rx.read(&mut buf[..want]).unwrap();
        p.push_with_meta("rf", InputData::Raw(&buf[..k]), &meta)
            .unwrap();
        got += k / 2;
        loop {
            let c = p.read_stream::<C32>("iq", &mut iq).unwrap();
            if c == 0 {
                break;
            }
            power.extend(iq[..c].iter().map(|v| f64::from(v.re * v.re + v.im * v.im)));
        }
        while p.read_stream::<f32>("audio", &mut audio).unwrap() > 0 {}
        while p.read_frame("spectrum", &mut frame).unwrap().is_some() {}
        drain_events(&mut p, &mut events);
    }

    // Measure: the first output sample above half the plateau, as a time.
    for (k, start_ns) in BURST_STARTS_NS.into_iter().enumerate() {
        let from = ((start_ns as f64 / OUTPUT_PERIOD_NS) as usize).saturating_sub(2000);
        let to = from + (BURST_SECONDS * 240_000.0) as usize + 4000;
        let window = &power[from..to];
        let mut plateau: Vec<f64> = window.to_vec();
        plateau.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let plateau = plateau[plateau.len() * 3 / 4];
        assert!(plateau > 0.3, "plateau {plateau}");
        let rise = window.iter().position(|&v| v > 0.5 * plateau).unwrap();
        let index = (from + rise) as u64;
        let measured = p
            .time_of("iq", index)
            .unwrap()
            .expect("an anchor was reported");
        let expected = T0_NS as f64 + start_ns as f64 + DELAY_SAMPLES / IF_RATE as f64 * 1e9;
        let error = measured as f64 - expected;
        println!("burst {k}: measured - expected = {error:.0} ns (limit {OUTPUT_PERIOD_NS:.0} ns)");
        assert!(
            error.abs() <= OUTPUT_PERIOD_NS,
            "burst {k}: measured {measured}, expected {expected}, error {error} ns"
        );
        // Before and after the burst there is only noise.
        assert!(window[..1000].iter().all(|&v| v < 0.01 * plateau));
    }
}
