//! The pipeline against hand-wired blocks: bit-exact output, block-size invariance, fan-out,
//! run-time parameters.
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

use sdrk::{C32, Rate};
use sdrk_blocks::{Block, ConvertBlock, ConvertConfig, FrameBlock, SpectrumBlock, SpectrumConfig};
use sdrk_device::formats;
use sdrk_pipeline::VecStorage;
use sdrk_testkit::ChunkPlan;
use serde_json::json;

use common::{
    HandChain, bits, build, cu8_recording, cu8_stations, flow, run_fm, run_fm_with, tone_power,
};

/// Half a second of the E2E-RX signal at 2.4 MHz.
const HALF_SECOND: usize = 1_200_000;

#[test]
fn pipeline_equals_the_hand_wired_chain() {
    let bytes = cu8_recording(HALF_SECOND);
    let mut pipeline = build(&flow("fm_receiver"));
    let (audio, frames) = run_fm(&mut pipeline, &bytes, 8192);

    let mut chain = HandChain::new();
    let mut expected = Vec::new();
    for piece in bytes.chunks(2000) {
        chain.feed(piece, &mut expected);
    }
    assert_eq!(audio.len(), expected.len(), "number of audio samples");
    assert!(audio.len() > 20_000, "half a second at 48 kHz");
    assert_eq!(bits(&audio), bits(&expected), "audio is bit-identical");
    assert!(!frames.is_empty());

    // The audio carries the 1 kHz tone of the station (after the filters have settled).
    let tail = &audio[audio.len() / 2..];
    let on = tone_power(tail, 1000.0, 48_000.0);
    let off = tone_power(tail, 3300.0, 48_000.0);
    assert!(on > 100.0 * off, "1 kHz tone: {on} vs {off}");
}

#[test]
fn chunking_does_not_change_the_output() {
    let bytes = cu8_recording(150_000);
    let mut reference = build(&flow("fm_receiver"));
    let (audio, frames) = run_fm(&mut reference, &bytes, 8192);
    assert!(!audio.is_empty());
    assert!(!frames.is_empty());
    for seed in 1..=10u64 {
        let mut pipeline = build(&flow("fm_receiver"));
        let plan = ChunkPlan::new(seed, 8192);
        let (a, f) = run_fm_with(&mut pipeline, &bytes, plan.sizes(), |_, _| {});
        assert_eq!(bits(&a), bits(&audio), "audio, seed {seed}");
        assert_eq!(f.len(), frames.len(), "frame count, seed {seed}");
        for (k, ((fa, ma), (fb, mb))) in f.iter().zip(&frames).enumerate() {
            assert_eq!(bits(fa), bits(fb), "frame {k}, seed {seed}");
            assert_eq!(ma, mb, "frame meta {k}, seed {seed}");
        }
    }
}

#[test]
fn fan_out_reads_the_same_buffer() {
    let bytes = cu8_recording(100_000);
    let mut pipeline = build(&flow("fm_receiver"));
    let (_, frames) = run_fm(&mut pipeline, &bytes, 3000);
    assert!(frames.len() >= 20);

    // A standalone spectrum block on the converted samples.
    let rate = Rate::hz(2_400_000).unwrap();
    let mut convert = ConvertBlock::<C32, _>::new(
        &ConvertConfig {
            format: formats::CU8,
        },
        rate,
        &mut VecStorage,
    )
    .unwrap();
    let cfg = SpectrumConfig {
        fft_size: 1024,
        averaging: 4,
        ..SpectrumConfig::default()
    };
    let mut spectrum = SpectrumBlock::<C32, _>::new(&cfg, rate, &mut VecStorage).unwrap();
    let mut iq = vec![C32::default(); bytes.len() / 2];
    let p = convert.process(&bytes, &mut iq).unwrap();
    assert_eq!(p.produced, iq.len());
    let mut expected = Vec::new();
    let mut rest = &iq[..];
    let mut frame = vec![0f32; 1024];
    while !rest.is_empty() {
        let (used, meta) = spectrum.push(rest, &mut frame).unwrap();
        rest = &rest[used..];
        if let Some(m) = meta {
            expected.push((frame.clone(), m));
        }
        if used == 0 {
            break;
        }
    }
    assert_eq!(frames.len(), expected.len());
    for (k, ((a, ma), (b, mb))) in frames.iter().zip(&expected).enumerate() {
        assert_eq!(bits(a), bits(b), "frame {k}");
        assert_eq!(ma, mb);
    }
}

#[test]
fn the_tuning_moves_the_demodulated_tone_from_the_next_chunk() {
    // Two stations: +600 kHz with a 1 kHz tone, -200 kHz with a 2 kHz tone.
    let bytes = cu8_stations(&[(600_000.0, 1000.0), (-200_000.0, 2000.0)], 1_200_000);
    let chunk = 8000usize; // bytes: 4000 samples
    let switch_at = 100usize; // chunk index of the change

    let mut pipeline = build(&flow("fm_receiver"));
    let mut audio_before = Vec::new();
    let mut audio_all = Vec::new();
    let mut buf = vec![0f32; 4096];
    let mut chain = HandChain::new();
    let mut expected = Vec::new();
    for (i, piece) in bytes.chunks(chunk).enumerate() {
        if i == switch_at {
            audio_before = audio_all.clone();
            pipeline
                .set_param("tune", "frequency_hz", json!(-200_000.0))
                .unwrap();
            chain
                .ddc
                .set_param("frequency_hz", sdrk_blocks::ParamValue::F64(-200_000.0))
                .unwrap();
        }
        pipeline
            .push("rf", sdrk_pipeline::InputData::Raw(piece))
            .unwrap();
        loop {
            let n = pipeline.read_stream("audio", &mut buf).unwrap();
            if n == 0 {
                break;
            }
            audio_all.extend_from_slice(&buf[..n]);
        }
        let mut frame = vec![0f32; 1024];
        while pipeline
            .read_frame("spectrum", &mut frame)
            .unwrap()
            .is_some()
        {}
        chain.feed(piece, &mut expected);
    }
    // Same samples as the hand-wired chain that got the same parameter change.
    assert_eq!(bits(&audio_all), bits(&expected));
    // The tone moved: 1 kHz before, 2 kHz after (allowing the filters to settle).
    let before = &audio_before[audio_before.len() / 2..];
    let after = &audio_all[audio_before.len() + 3000..];
    assert!(tone_power(before, 1000.0, 48_000.0) > 100.0 * tone_power(before, 2000.0, 48_000.0));
    assert!(tone_power(after, 2000.0, 48_000.0) > 100.0 * tone_power(after, 1000.0, 48_000.0));
    // And the readout shows the change.
    let cfg = pipeline.config();
    assert_eq!(cfg.nodes["tune"].params["frequency_hz"], json!(-200_000.0));
}
