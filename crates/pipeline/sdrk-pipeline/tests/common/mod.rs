//! Helpers shared by the integration tests.
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
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
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::redundant_closure_for_method_calls
)]

use std::path::PathBuf;

use num_complex::Complex64;
use sdrk::{C32, Rate};
use sdrk_blocks::{
    Block, ConvertBlock, ConvertConfig, DdcBlock, DdcConfig, FmDemodBlock, FmDemodConfig,
};
use sdrk_device::formats;
use sdrk_pipeline::{BuildOptions, InputData, Pipeline, Registry, VecStorage};
use sdrk_testkit::e2e::{
    AUDIO_HZ, CARRIER_OFFSET_HZ, CHUNK, DEVIATION_HZ, IF_RATE_HZ, RX_SIGNAL_AMPLITUDE,
};
use sdrk_testkit::{encode::encode, fm::fm_baseband};

/// Path of a file below the workspace root.
pub fn workspace_file(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join(rel)
}

/// Text of an example flow.
pub fn flow(name: &str) -> String {
    std::fs::read_to_string(workspace_file(&format!("examples/flows/{name}.json"))).unwrap()
}

/// Builds a pipeline from JSON text with the default options.
pub fn build(json: &str) -> Pipeline {
    Registry::standard()
        .build_json(json, &BuildOptions::default())
        .unwrap_or_else(|e| panic!("build failed: {e}"))
}

/// CU8 bytes of `stations` FM broadcasts (`(carrier offset, audio frequency)`), `samples` long,
/// generated with the independent testkit encoder.
pub fn cu8_stations(stations: &[(f64, f64)], samples: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; 2 * samples];
    let mut block = vec![Complex64::new(0.0, 0.0); CHUNK];
    let mut one = vec![Complex64::new(0.0, 0.0); CHUNK];
    let amp = RX_SIGNAL_AMPLITUDE / stations.len() as f64;
    let mut start = 0usize;
    while start < samples {
        let n = CHUNK.min(samples - start);
        block[..n].fill(Complex64::new(0.0, 0.0));
        for &(offset, audio) in stations {
            fm_baseband(
                &mut one[..n],
                IF_RATE_HZ as f64,
                offset,
                DEVIATION_HZ,
                audio,
                1.0,
                start as u64,
            );
            for (b, o) in block[..n].iter_mut().zip(&one[..n]) {
                *b += *o * amp;
            }
        }
        encode(
            &formats::CU8,
            &block[..n],
            &mut bytes[2 * start..2 * (start + n)],
        )
        .unwrap();
        start += n;
    }
    bytes
}

/// CU8 bytes of the E2E-RX signal (one station at +600 kHz, 1 kHz audio).
pub fn cu8_recording(samples: usize) -> Vec<u8> {
    cu8_stations(&[(CARRIER_OFFSET_HZ, AUDIO_HZ)], samples)
}

/// Pushes `bytes` in pieces of `chunk` bytes and reads all audio and spectrum frames after
/// every push.
pub fn run_fm(
    p: &mut Pipeline,
    bytes: &[u8],
    chunk: usize,
) -> (Vec<f32>, Vec<(Vec<f32>, sdrk_blocks::FrameMeta)>) {
    run_fm_with(p, bytes, std::iter::repeat(chunk), |_, _| {})
}

/// Like [`run_fm`] with chunk sizes from an iterator and a hook before every push.
pub fn run_fm_with(
    p: &mut Pipeline,
    bytes: &[u8],
    mut sizes: impl Iterator<Item = usize>,
    mut before: impl FnMut(usize, &mut Pipeline),
) -> (Vec<f32>, Vec<(Vec<f32>, sdrk_blocks::FrameMeta)>) {
    let mut audio = Vec::new();
    let mut frames = Vec::new();
    let mut buf = vec![0f32; 4096];
    let mut frame = vec![0f32; 1024];
    let mut pos = 0;
    let mut index = 0;
    while pos < bytes.len() {
        let n = sizes.next().unwrap().min(bytes.len() - pos);
        before(index, p);
        p.push("rf", InputData::Raw(&bytes[pos..pos + n])).unwrap();
        pos += n;
        index += 1;
        loop {
            let got = p.read_stream("audio", &mut buf).unwrap();
            if got == 0 {
                break;
            }
            audio.extend_from_slice(&buf[..got]);
        }
        while let Some(meta) = p.read_frame("spectrum", &mut frame).unwrap() {
            frames.push((frame.clone(), meta));
        }
    }
    (audio, frames)
}

/// The receive chain of `fm_receiver.json` built by hand from the blocks.
pub struct HandChain {
    pub convert: ConvertBlock<C32, VecStorage>,
    pub ddc: DdcBlock<C32, VecStorage>,
    pub demod: FmDemodBlock<C32, VecStorage>,
}

impl HandChain {
    pub fn new() -> Self {
        let rate = Rate::hz(2_400_000).unwrap();
        let convert = ConvertBlock::<C32, _>::new(
            &ConvertConfig {
                format: formats::CU8,
            },
            rate,
            &mut VecStorage,
        )
        .unwrap();
        let ddc = DdcBlock::<C32, _>::new(
            &DdcConfig {
                frequency_hz: 600_000.0,
                decimation: 10,
                cutoff_hz: Some(100_000.0),
                ..DdcConfig::default()
            },
            rate,
            &mut VecStorage,
        )
        .unwrap();
        let demod = FmDemodBlock::<C32, _>::new(
            &FmDemodConfig {
                deviation_hz: 75_000.0,
                deemphasis_us: 50.0,
                audio_decimation: 5,
                ..FmDemodConfig::default()
            },
            Rate::hz(240_000).unwrap(),
            &mut VecStorage,
        )
        .unwrap();
        Self {
            convert,
            ddc,
            demod,
        }
    }

    /// Feeds one piece of bytes through the three blocks and appends the audio.
    pub fn feed(&mut self, bytes: &[u8], audio: &mut Vec<f32>) {
        let mut iq = vec![C32::default(); bytes.len() / 2 + 8];
        let mut bb = vec![C32::default(); bytes.len() / 20 + 64];
        let mut out = vec![0f32; bytes.len() / 100 + 64];
        let mut rest = bytes;
        while !rest.is_empty() {
            let p = self.convert.process(rest, &mut iq).unwrap();
            if p.consumed == 0 {
                break;
            }
            rest = &rest[p.consumed..];
            let mut samples = &iq[..p.produced];
            while !samples.is_empty() {
                let q = self.ddc.process(samples, &mut bb).unwrap();
                samples = &samples[q.consumed..];
                let mut low = &bb[..q.produced];
                while !low.is_empty() {
                    let r = self.demod.process(low, &mut out).unwrap();
                    audio.extend_from_slice(&out[..r.produced]);
                    low = &low[r.consumed..];
                    if r.consumed == 0 && r.produced == 0 {
                        break;
                    }
                }
                if q.consumed == 0 && q.produced == 0 {
                    break;
                }
            }
        }
    }
}

/// Power of the frequency `f` in `x` at the sample rate `fs` (Goertzel).
pub fn tone_power(x: &[f32], f: f64, fs: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * f / fs;
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (n, v) in x.iter().enumerate() {
        re += f64::from(*v) * (w * n as f64).cos();
        im += f64::from(*v) * (w * n as f64).sin();
    }
    (re * re + im * im) / (x.len() as f64).powi(2)
}

/// Bit patterns of a float slice.
pub fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}
