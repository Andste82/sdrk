//! End-to-end noise reduction (M33): E2E-NB (`e2e_noise_blank_f32`, `e2e_noise_blank_q15`) and
//! E2E-NOTCH (`e2e_auto_notch_f32`) on the receive chain of the example flow
//! `examples/flows/hf_am_receiver.json`, built from blocks.
//!
//! The chain: `ConvertBlock` (`cs16` at 192 kHz), `NoiseBlankBlock` (lookahead 10 us, no
//! hold), `DdcBlock`
//! (decimation 4, cutoff 4500 Hz), `SquelchBlock` (the defaults), `AgcBlock` (attack 0.1 s,
//! release 0.5 s), `AmDemodBlock` (the parameters of the flow) and `AutoNotchRealBlock` (the
//! defaults). The measured values are printed (run with `--nocapture` to see them).
//!
//! `the_chain_is_the_flow` (feature `std`) parses the flow file and checks that its nodes,
//! parameters and edges are exactly this chain, so the measured values hold for the flow.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::print_stderr,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names
)]

use core::f64::consts::PI;

use num_complex::Complex64;
use sdrk::{Rate, Sdrk};
use sdrk_blocks::{
    AgcBlock, AgcConfig, AmDemodBlock, AmDemodConfig, AutoNotchConfig, AutoNotchRealBlock, Block,
    BlockSample, ConvertBlock, ConvertConfig, DdcBlock, DdcConfig, NoiseBlankBlock,
    NoiseBlankConfig, SquelchBlock, SquelchConfig, StreamElem,
};
use sdrk_core::kernels::agc::Agc;
use sdrk_core::kernels::convert::Convert;
use sdrk_core::kernels::decim::ResampleReal;
use sdrk_core::kernels::demod::DcBlockReal;
use sdrk_core::kernels::fused::Ddc;
use sdrk_core::kernels::noise_blank::NoiseBlank;
use sdrk_core::kernels::spectrum::Mag;
use sdrk_core::kernels::squelch::Squelch;
use sdrk_core::{C32, CQ15, ComplexSample, RealSample};
use sdrk_device::formats;
use sdrk_testkit::encode::encode;
use sdrk_testkit::meters::ToneAccumulator;
use sdrk_testkit::{Pcg32, VecStorage};

/// Input rate of the flow in Hz.
const FS: u64 = 192_000;
/// Decimation of the channel filter (192 kHz to 48 kHz).
const DECIMATION: usize = 4;
/// Cutoff of the channel filter in Hz.
const CUTOFF_HZ: f64 = 4_500.0;
/// Decimation of the audio in `am_demod` (48 kHz to the audio rate of the flow).
const AUDIO_DECIMATION: usize = 4;
/// Cutoff of the audio low-pass of `am_demod` in Hz.
const AUDIO_CUTOFF_HZ: f64 = 4_000.0;
/// Audio rate of the flow in Hz.
const AUDIO_RATE: f64 = (FS as usize / DECIMATION / AUDIO_DECIMATION) as f64;
/// Lookahead of the blanker of the flow in seconds (D373).
const NB_LOOKAHEAD_S: f64 = 0.00001;
/// Hold of the blanker of the flow in seconds (D373).
const NB_HOLD_S: f64 = 0.0;
/// Attack time of the AGC of the flow in seconds.
const AGC_ATTACK_S: f64 = 0.1;
/// Release time of the AGC of the flow in seconds.
const AGC_RELEASE_S: f64 = 0.5;
/// Offset of the station from the tuner centre in Hz (the `ddc` is tuned to it).
const OFFSET_HZ: f64 = 12_000.0;

/// E2E-NB: the AM test signal of M20 (1 kHz tone, modulation index 0.5, noise sigma 0.01 per
/// component at carrier 0.5), scaled by 0.08 so that the impulses of 20 times the carrier
/// stay inside the full scale of `cs16`: carrier 0.04, noise 0.0008, impulses 0.8.
const NB_SCALE: f64 = 0.08;
/// Seed of E2E-NB.
const NB_SEED: u64 = 476;
/// Seed of E2E-NOTCH.
const NOTCH_SEED: u64 = 477;
/// Length of E2E-NB in input samples (3 s).
const NB_SAMPLES: usize = 3 * FS as usize;
/// Length of E2E-NOTCH in input samples (2^20, about 5.5 s; a power of two for the FFT of the
/// modulation).
const NOTCH_SAMPLES: usize = 1 << 20;

/// The receive chain of `hf_am_receiver`; `nb` and `notch` switch the two M33 blocks.
struct Chain<T: BlockSample> {
    convert: ConvertBlock<T, VecStorage>,
    nb: Option<NoiseBlankBlock<T, VecStorage>>,
    ddc: DdcBlock<T, VecStorage>,
    squelch: SquelchBlock<T, VecStorage>,
    agc: AgcBlock<T, VecStorage>,
    demod: AmDemodBlock<T, VecStorage>,
}

/// The blanker of the flow: the defaults with a lookahead of 10 us (2 samples at 192 kHz) and
/// no hold. Each blanked sample leaves a hole in the carrier that the channel filter turns into
/// a dip of the envelope, so the blanking must be as short as the impulses allow (D373).
fn nb_config() -> NoiseBlankConfig {
    NoiseBlankConfig {
        lookahead_s: NB_LOOKAHEAD_S,
        hold_s: NB_HOLD_S,
        ..NoiseBlankConfig::default()
    }
}

fn ddc_config() -> DdcConfig {
    DdcConfig {
        frequency_hz: OFFSET_HZ,
        decimation: DECIMATION,
        cutoff_hz: Some(CUTOFF_HZ),
        ..DdcConfig::default()
    }
}

fn demod_config() -> AmDemodConfig {
    AmDemodConfig {
        audio_decimation: AUDIO_DECIMATION,
        audio_cutoff_hz: AUDIO_CUTOFF_HZ,
        ..AmDemodConfig::default()
    }
}

/// Runs a block over `input` and resubmits what it did not consume.
fn pump<I: StreamElem, O: StreamElem + Default, B: Block<In = I, Out = O>>(
    block: &mut B,
    input: &[I],
) -> Vec<O> {
    let mut out = Vec::with_capacity(input.len());
    let mut room = vec![O::default(); 8192];
    let mut rest = input;
    loop {
        let p = block.process(rest, &mut room).unwrap();
        out.extend_from_slice(&room[..p.produced]);
        rest = &rest[p.consumed..];
        if rest.is_empty() && p.produced < room.len() {
            break;
        }
        assert!(p.consumed > 0 || p.produced > 0, "no progress");
    }
    out
}

impl<T: BlockSample> Chain<T>
where
    Sdrk: Convert<T>
        + NoiseBlank<T>
        + Ddc<T>
        + Squelch<T>
        + Agc<T>
        + Mag<T>
        + DcBlockReal<<T as ComplexSample>::Real>
        + ResampleReal<<T as ComplexSample>::Real>,
{
    fn new(nb: bool) -> Self {
        let fs = Rate::hz(FS).unwrap();
        let ch = Rate::hz(FS / DECIMATION as u64).unwrap();
        let s = &mut VecStorage;
        let cfg = ConvertConfig {
            format: formats::CS16,
        };
        Self {
            convert: ConvertBlock::new(&cfg, fs, s).unwrap(),
            nb: nb.then(|| NoiseBlankBlock::new(&nb_config(), fs, s).unwrap()),
            ddc: DdcBlock::new(&ddc_config(), fs, s).unwrap(),
            squelch: SquelchBlock::new(&SquelchConfig::default(), ch, s).unwrap(),
            agc: AgcBlock::new(
                &AgcConfig {
                    attack_s: AGC_ATTACK_S,
                    release_s: AGC_RELEASE_S,
                    ..AgcConfig::default()
                },
                ch,
                s,
            )
            .unwrap(),
            demod: AmDemodBlock::new(&demod_config(), ch, s).unwrap(),
        }
    }

    /// The audio of `cs16` bytes, as f64.
    fn run(&mut self, bytes: &[u8]) -> Vec<f64> {
        self.run_raw(bytes).iter().map(|v| v.to_f64()).collect()
    }

    /// The audio as raw sample values (for the bit-exact comparison).
    fn run_raw(&mut self, bytes: &[u8]) -> Vec<<T as ComplexSample>::Real> {
        let iq: Vec<T> = pump(&mut self.convert, bytes);
        let iq = match &mut self.nb {
            Some(nb) => pump(nb, &iq),
            None => iq,
        };
        let low = pump(&mut self.ddc, &iq);
        let gated = pump(&mut self.squelch, &low);
        let leveled = pump(&mut self.agc, &gated);
        pump(&mut self.demod, &leveled)
    }
}

/// `x` shifted from baseband to the offset of the station.
fn shift(x: &mut [Complex64]) {
    for (n, v) in x.iter_mut().enumerate() {
        let cycles = OFFSET_HZ * n as f64 / FS as f64;
        let w = 2.0 * PI * (cycles - cycles.floor());
        *v *= Complex64::new(w.cos(), w.sin());
    }
}

fn to_cs16(x: &[Complex64]) -> Vec<u8> {
    let mut bytes = vec![0u8; formats::CS16.bytes_for_samples(x.len()).unwrap()];
    encode(&formats::CS16, x, &mut bytes).unwrap();
    bytes
}

/// The E2E-NB input: the scaled AM signal at the offset plus noise, and the same with the
/// impulses (3 samples of amplitude 20 times the carrier with a random phase, 100 per second:
/// one at a random position in every slot of 1920 samples).
fn nb_input() -> (Vec<Complex64>, Vec<Complex64>, usize) {
    let carrier = 0.5 * NB_SCALE;
    let sigma = 0.01 * NB_SCALE;
    let mut noise = Pcg32::new(NB_SEED, 1);
    let mut clean: Vec<Complex64> = (0..NB_SAMPLES)
        .map(|n| {
            let cycles = 1_000.0 * n as f64 / FS as f64;
            let env = carrier * (1.0 + 0.5 * (2.0 * PI * (cycles - cycles.floor())).cos());
            Complex64::new(env, 0.0)
        })
        .collect();
    shift(&mut clean);
    for v in &mut clean {
        v.re += sigma * noise.normal();
        v.im += sigma * noise.normal();
    }
    let mut hits = Pcg32::new(NB_SEED, 2);
    let mut impulsive = clean.clone();
    let slot = FS as usize / 100;
    let mut count = 0;
    for start in (0..NB_SAMPLES).step_by(slot) {
        let at = start + hits.below((slot - 3) as u32) as usize;
        let phase = 2.0 * PI * hits.next_f64();
        let p = Complex64::new(phase.cos(), phase.sin()) * (20.0 * carrier);
        for v in &mut impulsive[at..at + 3] {
            *v = p;
        }
        count += 1;
    }
    (clean, impulsive, count)
}

/// SINAD at 1 kHz over the audio after the first second (the AGC has settled).
fn sinad_1k(audio: &[f64]) -> f64 {
    let mut acc = ToneAccumulator::new(AUDIO_RATE, 1_000.0);
    for &v in &audio[AUDIO_RATE as usize..] {
        acc.push(v);
    }
    acc.sinad_db()
}

fn e2e_noise_blank<T: BlockSample>(name: &str)
where
    Sdrk: Convert<T>
        + NoiseBlank<T>
        + Ddc<T>
        + Squelch<T>
        + Agc<T>
        + Mag<T>
        + DcBlockReal<<T as ComplexSample>::Real>
        + ResampleReal<<T as ComplexSample>::Real>,
    <T as ComplexSample>::Real: PartialEq + core::fmt::Debug,
{
    let (clean, impulsive, count) = nb_input();
    let bytes = to_cs16(&impulsive);
    let with = sinad_1k(&Chain::<T>::new(true).run(&bytes));
    let without = sinad_1k(&Chain::<T>::new(false).run(&bytes));
    let clean_sinad = sinad_1k(&Chain::<T>::new(false).run(&to_cs16(&clean)));
    let mut chain = Chain::<T>::new(true);
    chain.run(&bytes);
    let blanked = chain.nb.as_ref().unwrap().blanked();
    eprintln!(
        "{name}: {count} impulses, {blanked} samples blanked; SINAD at 1 kHz with \
         noise_blank {with:.2} dB, without {without:.2} dB, improvement {:.2} dB (without \
         impulses {clean_sinad:.2} dB)",
        with - without
    );
    assert!(
        with >= without + 10.0,
        "{name}: SINAD with noise_blank {with:.2} dB, without {without:.2} dB: \
         the improvement is below 10 dB"
    );

    // Without impulses the blanker changes nothing but the delay: the chain without it, fed
    // with the input delayed by D samples, gives the same audio bit for bit.
    let mut chain = Chain::<T>::new(true);
    let d = chain.nb.as_ref().unwrap().lookahead();
    assert_eq!(d, 2);
    let blanked_audio = chain.run_raw(&to_cs16(&clean));
    assert_eq!(chain.nb.as_ref().unwrap().blanked(), 0, "{name}: blanked");
    let mut delayed = vec![Complex64::new(0.0, 0.0); d];
    delayed.extend_from_slice(&clean[..clean.len() - d]);
    let plain_audio = Chain::<T>::new(false).run_raw(&to_cs16(&delayed));
    assert_eq!(blanked_audio.len(), plain_audio.len());
    let first = blanked_audio
        .iter()
        .zip(&plain_audio)
        .position(|(a, b)| a.to_f64().to_bits() != b.to_f64().to_bits());
    assert_eq!(first, None, "{name}: the clean audio differs");
    eprintln!(
        "{name}: clean input, {} audio samples identical with and without noise_blank \
         (aligned by D = {d})",
        plain_audio.len()
    );
}

#[test]
fn e2e_noise_blank_f32() {
    e2e_noise_blank::<C32>("e2e_noise_blank_f32");
}

#[test]
fn e2e_noise_blank_q15() {
    e2e_noise_blank::<CQ15>("e2e_noise_blank_q15");
}

// ---- E2E-NOTCH ---------------------------------------------------------------------------

/// In-place radix-2 FFT (`inverse` without the 1/n scale).
fn fft(x: &mut [Complex64], inverse: bool) {
    let n = x.len();
    assert!(n.is_power_of_two());
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            x.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let w = 2.0 * PI / len as f64 * sign;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let t = Complex64::new((w * k as f64).cos(), (w * k as f64).sin())
                    * x[start + k + len / 2];
                let u = x[start + k];
                x[start + k] = u + t;
                x[start + k + len / 2] = u - t;
            }
        }
        len <<= 1;
    }
}

/// Speech-like modulation: Gaussian noise (seed 477) limited to 300 to 3000 Hz in the frequency
/// domain, scaled to an RMS of 0.2 and clipped to +-0.95 (no overmodulation).
fn speech_like(n: usize) -> Vec<f64> {
    let mut rng = Pcg32::new(NOTCH_SEED, 1);
    let mut spec = vec![Complex64::new(0.0, 0.0); n];
    let df = FS as f64 / n as f64;
    for (k, v) in spec.iter_mut().enumerate().take(n / 2).skip(1) {
        let f = k as f64 * df;
        if (300.0..=3000.0).contains(&f) {
            *v = Complex64::new(rng.normal(), rng.normal());
        }
    }
    for k in 1..n / 2 {
        spec[n - k] = spec[k].conj();
    }
    fft(&mut spec, true);
    let rms = (spec.iter().map(|v| v.re * v.re).sum::<f64>() / n as f64).sqrt();
    spec.iter()
        .map(|v| (0.2 * v.re / rms).clamp(-0.95, 0.95))
        .collect()
}

/// The audio of the f32 chain for the E2E-NOTCH input, with or without the notch.
fn notch_audio(bytes: &[u8], notch: bool) -> Vec<f64> {
    let audio = Chain::<C32>::new(true).run(bytes);
    if !notch {
        return audio;
    }
    let x: Vec<f32> = audio.iter().map(|v| *v as f32).collect();
    let mut block = AutoNotchRealBlock::new(
        &AutoNotchConfig::default(),
        Rate::hz(AUDIO_RATE as u64).unwrap(),
        &mut VecStorage,
    )
    .unwrap();
    pump(&mut block, &x).iter().map(|v| f64::from(*v)).collect()
}

/// Mean power per bin of `x` (a power-of-two prefix, Hann window) between `lo` and `hi` Hz,
/// leaving out `|f - skip| < 25 Hz`.
fn band_power(x: &[f64], lo: f64, hi: f64, skip: f64) -> f64 {
    let n = 1 << (usize::BITS - 1 - x.len().leading_zeros());
    let mut s: Vec<Complex64> = x[..n]
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let w = 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos();
            Complex64::new(v * w, 0.0)
        })
        .collect();
    fft(&mut s, false);
    let df = AUDIO_RATE / n as f64;
    s.iter()
        .take(n / 2)
        .enumerate()
        .filter(|(k, _)| {
            let f = *k as f64 * df;
            f >= lo && f <= hi && (f - skip).abs() >= 25.0
        })
        .map(|(_, v)| v.norm_sqr())
        .sum()
}

#[test]
fn e2e_auto_notch_f32() {
    let m = speech_like(NOTCH_SAMPLES);
    let mut x: Vec<Complex64> = m
        .iter()
        .map(|v| Complex64::new(0.5 * (1.0 + v), 0.0))
        .collect();
    // The interfering carrier 1.5 kHz above the station, amplitude 0.3.
    for (n, v) in x.iter_mut().enumerate() {
        let cycles = 1_500.0 * n as f64 / FS as f64;
        let w = 2.0 * PI * (cycles - cycles.floor());
        *v += Complex64::new(0.3 * w.cos(), 0.3 * w.sin());
    }
    shift(&mut x);
    let mut noise = Pcg32::new(NOTCH_SEED, 2);
    for v in &mut x {
        v.re += 0.01 * noise.normal();
        v.im += 0.01 * noise.normal();
    }
    let bytes = to_cs16(&x);
    let with = notch_audio(&bytes, true);
    let without = notch_audio(&bytes, false);
    assert_eq!(with.len(), without.len());
    // The whistle after 0.5 s.
    let from = (AUDIO_RATE / 2.0) as usize;
    let tone = |a: &[f64]| {
        let mut acc = ToneAccumulator::new(AUDIO_RATE, 1_500.0);
        for &v in a {
            acc.push(v);
        }
        acc.amplitude()
    };
    let (a_with, a_without) = (tone(&with[from..]), tone(&without[from..]));
    let attenuation = 20.0 * (a_without / a_with).log10();
    // The modulation in 300 to 3000 Hz, without the whistle.
    let (p_with, p_without) = (
        band_power(&with[from..], 300.0, 3000.0, 1_500.0),
        band_power(&without[from..], 300.0, 3000.0, 1_500.0),
    );
    let level_change = 10.0 * (p_with / p_without).log10();
    eprintln!(
        "e2e_auto_notch_f32: whistle at 1.5 kHz {a_without:.5} without, {a_with:.6} with \
         auto_notch: attenuation {attenuation:.2} dB; audio level in 300 to 3000 Hz \
         (whistle excluded) changes by {level_change:.2} dB"
    );
    assert!(
        attenuation >= 30.0,
        "the whistle is attenuated by {attenuation:.2} dB, need 30 dB"
    );
    assert!(
        level_change.abs() <= 2.0,
        "the audio level changes by {level_change:.2} dB, at most 2 dB allowed"
    );
}

/// The flow file `examples/flows/hf_am_receiver.json` (its YAML twin is checked equal to it by
/// `sdrk-pipeline/tests/yaml.rs`) is exactly the chain of this file: the same nodes with the
/// same parameters (the others at their defaults), in the same order. E2E-NB and E2E-NOTCH
/// therefore measure the flow; a change of the flow must change this file too.
#[cfg(feature = "std")]
#[test]
fn the_chain_is_the_flow() {
    use serde_json::{Value, json};

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../examples/flows/hf_am_receiver.json"
    );
    let flow: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(flow["settings"]["precision"], json!("f32"));
    assert_eq!(flow["inputs"]["rf"]["format"], json!("cs16"));
    assert_eq!(flow["inputs"]["rf"]["rate_hz"], json!(FS));
    let params = |node: &str, kind: &str| -> Value {
        let n = &flow["nodes"][node];
        assert_eq!(n["kind"], json!(kind), "{node}");
        n.get("params").cloned().unwrap_or_else(|| json!({}))
    };
    let num = |v: &Value| v.as_f64().unwrap();
    // Every parameter is listed, so a parameter added to the flow fails here.
    let nb = params("nb", "noise_blank");
    assert_eq!(nb.as_object().unwrap().len(), 2);
    assert_eq!(num(&nb["lookahead_s"]), NB_LOOKAHEAD_S);
    assert_eq!(num(&nb["hold_s"]), NB_HOLD_S);
    let tune = params("tune", "ddc");
    assert_eq!(tune.as_object().unwrap().len(), 2);
    assert_eq!(tune["decimation"].as_u64().unwrap(), DECIMATION as u64);
    assert_eq!(num(&tune["cutoff_hz"]), CUTOFF_HZ);
    let agc = params("agc", "agc");
    assert_eq!(agc.as_object().unwrap().len(), 2);
    assert_eq!(num(&agc["attack_s"]), AGC_ATTACK_S);
    assert_eq!(num(&agc["release_s"]), AGC_RELEASE_S);
    let demod = params("demod", "am_demod");
    assert_eq!(demod.as_object().unwrap().len(), 2);
    assert_eq!(
        demod["audio_decimation"].as_u64().unwrap(),
        AUDIO_DECIMATION as u64
    );
    assert_eq!(num(&demod["audio_cutoff_hz"]), AUDIO_CUTOFF_HZ);
    assert_eq!(params("squelch", "squelch"), json!({}));
    assert_eq!(params("notch", "auto_notch"), json!({}));
    assert_eq!(flow["nodes"].as_object().unwrap().len(), 6);
    let edges: Vec<(&str, &str)> = flow["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| (e["from"].as_str().unwrap(), e["to"].as_str().unwrap()))
        .collect();
    assert_eq!(
        edges,
        [
            ("rf", "nb"),
            ("nb", "tune"),
            ("tune", "squelch"),
            ("squelch", "agc"),
            ("agc", "demod"),
            ("demod", "notch"),
            ("notch", "audio"),
        ]
    );
}
