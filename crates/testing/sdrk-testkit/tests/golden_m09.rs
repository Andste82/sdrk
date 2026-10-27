//! Golden tests of M09: filter design, the `dot` kernels and the blocks.
//!
//! The blocks are driven in one piece and in chunks of random size; both runs must be
//! bit-identical and match the golden file.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::cast_lossless
)]

use sdrk::Sdrk;
use sdrk_blocks::design::{DesignWindow, firwin_lowpass, kaiserord};
use sdrk_blocks::{
    AmDemodBlock, AmDemodConfig, Block, DdcBlock, DdcConfig, FmDemodBlock, FmDemodConfig,
    FrameBlock, ResamplerBlock, ResamplerConfig, SpectrumBlock, SpectrumConfig, SpectrumWindow,
    SsbDemodBlock, SsbDemodConfig, SsbSideband,
};
use sdrk_core::kernels::dot::{Dot, DotReal};
use sdrk_core::{C32, CQ15, Q15, Rate, Reference, Sample};
use sdrk_testkit::golden::{self, Array, GoldenCase, Tolerance};
use sdrk_testkit::{ChunkPlan, VecStorage, bit_exact};

fn f64_array(case: &GoldenCase, name: &str) -> Vec<f64> {
    match case.array(name).unwrap() {
        Array::F64(v) => v.clone(),
        other => panic!("{name} must be f64, got {}", other.dtype()),
    }
}

fn max_abs_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn tolerance_limit(case: &GoldenCase) -> f64 {
    match case.tolerance {
        Tolerance::MaxAbsErr { max, .. } => max,
        ref other => panic!("expected a max_abs_err tolerance, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// design

#[test]
fn design_cases_are_present() {
    let names = golden::list("design").unwrap();
    for w in ["hann", "hamming", "blackman"] {
        for n in [31, 128] {
            assert!(names.contains(&format!("firwin_{w}_{n}")));
        }
    }
    for tag in ["0p5", "5p0", "8p6"] {
        assert!(names.contains(&format!("firwin_kaiser_{tag}")));
    }
    assert!(names.contains(&"kaiserord_table".to_string()));
    // The biquad design cases (M22) have their own test file.
    let m09 = names
        .iter()
        .filter(|n| !n.starts_with("biquad_") && !n.starts_with("butterworth_"))
        .count();
    assert_eq!(m09, 10);
}

#[test]
fn firwin_matches_scipy() {
    for name in golden::list("design").unwrap() {
        if !name.starts_with("firwin_") {
            continue;
        }
        let case = golden::load("design", &name).unwrap();
        let cutoff = case.param_f64("cutoff_hz").unwrap();
        let fs = case.param_f64("fs_hz").unwrap();
        let n = usize::try_from(case.param_u64("num_taps").unwrap()).unwrap();
        let window = match case.param_str("window").unwrap() {
            "hann" => DesignWindow::Hann,
            "hamming" => DesignWindow::Hamming,
            "blackman" => DesignWindow::Blackman,
            "kaiser" => DesignWindow::Kaiser {
                beta: case.param_f64("beta").unwrap(),
            },
            other => panic!("unknown window {other}"),
        };
        let mut taps = vec![0.0; n];
        firwin_lowpass(cutoff, fs, window, &mut taps).unwrap();
        let err = max_abs_err(&taps, &f64_array(&case, "expected"));
        assert!(err <= tolerance_limit(&case), "{name}: largest error {err}");
    }
}

#[test]
fn kaiserord_matches_scipy() {
    let case = golden::load("design", "kaiserord_table").unwrap();
    let ripple = f64_array(&case, "ripple_db");
    let width = f64_array(&case, "width");
    let Array::I32(taps) = case.array("num_taps").unwrap() else {
        panic!("num_taps must be i32");
    };
    let beta = f64_array(&case, "expected");
    let fs = case.param_f64("fs_hz").unwrap();
    assert_eq!(ripple.len(), 20);
    let mut got_beta = Vec::new();
    for ((a, w), n) in ripple.iter().zip(&width).zip(taps) {
        let (num, b) = kaiserord(*a, *w, fs).unwrap();
        assert_eq!(num, usize::try_from(*n).unwrap(), "ripple {a}, width {w}");
        got_beta.push(b);
    }
    assert!(max_abs_err(&got_beta, &beta) <= tolerance_limit(&case));
}

// ---------------------------------------------------------------------------------------------
// dot

fn rows(case: &GoldenCase) -> usize {
    usize::try_from(case.param_u64("rows").unwrap()).unwrap()
}

fn taps_len(case: &GoldenCase) -> usize {
    usize::try_from(case.param_u64("taps").unwrap()).unwrap()
}

#[test]
fn dot_golden() {
    let names = golden::list("dot").unwrap();
    assert_eq!(names.len(), 10);
    for name in names {
        let case = golden::load("dot", &name).unwrap();
        let (n, r) = (taps_len(&case), rows(&case));
        match (
            name.split('_').next().unwrap(),
            case.array("input").unwrap(),
        ) {
            ("c32", Array::CF32(x)) => {
                let Array::F32(t) = case.array("taps").unwrap() else {
                    panic!()
                };
                let expected = case.array("expected").unwrap();
                for backend in 0..2 {
                    let got: Vec<C32> = (0..r)
                        .map(|i| {
                            if backend == 0 {
                                Dot::<C32>::dot(&Sdrk, &x[i..i + n], t).unwrap()
                            } else {
                                Dot::<C32>::dot(&Reference, &x[i..i + n], t).unwrap()
                            }
                        })
                        .collect();
                    golden::assert_matches(&case, expected, &got).unwrap();
                }
            }
            ("cq15", Array::CI16(x)) => {
                let t = case.array("taps").unwrap().as_q15().unwrap();
                let expected = case.array("expected").unwrap();
                for backend in 0..2 {
                    let got: Vec<CQ15> = (0..r)
                        .map(|i| {
                            if backend == 0 {
                                Dot::<CQ15>::dot(&Sdrk, &x[i..i + n], &t).unwrap()
                            } else {
                                Dot::<CQ15>::dot(&Reference, &x[i..i + n], &t).unwrap()
                            }
                        })
                        .collect();
                    golden::assert_matches(&case, expected, &got).unwrap();
                }
            }
            ("f32", Array::F32(x)) => {
                let Array::F32(t) = case.array("taps").unwrap() else {
                    panic!()
                };
                let expected = case.array("expected").unwrap();
                for backend in 0..2 {
                    let got: Vec<f32> = (0..r)
                        .map(|i| {
                            if backend == 0 {
                                DotReal::<f32>::dot_real(&Sdrk, &x[i..i + n], t).unwrap()
                            } else {
                                DotReal::<f32>::dot_real(&Reference, &x[i..i + n], t).unwrap()
                            }
                        })
                        .collect();
                    golden::assert_matches(&case, expected, &got).unwrap();
                }
            }
            ("q15", Array::I16(x)) => {
                let x: Vec<Q15> = x.iter().map(|&v| Q15(v)).collect();
                let t = case.array("taps").unwrap().as_q15().unwrap();
                let expected = case.array("expected").unwrap();
                // The expected values are i16; compare through the Q15 view.
                for backend in 0..2 {
                    let got: Vec<Q15> = (0..r)
                        .map(|i| {
                            if backend == 0 {
                                DotReal::<Q15>::dot_real(&Sdrk, &x[i..i + n], &t).unwrap()
                            } else {
                                DotReal::<Q15>::dot_real(&Reference, &x[i..i + n], &t).unwrap()
                            }
                        })
                        .collect();
                    golden::assert_matches(&case, expected, &got).unwrap();
                }
            }
            (kind, other) => panic!("{name}: unexpected {kind} input {}", other.dtype()),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// block drivers

/// Feeds `input` to the block in pieces from `plan` (all at once for `None`), resubmitting what
/// the block did not consume together with the next piece.
fn run_block<B: Block>(block: &mut B, input: &[B::In], plan: Option<ChunkPlan>) -> Vec<B::Out>
where
    B::Out: Default,
{
    let mut out = Vec::new();
    let mut room = vec![B::Out::default(); 2048];
    let mut pending: Vec<B::In> = Vec::new();
    let mut pos = 0;
    let mut sizes = plan.map(|p| p.sizes());
    while pos < input.len() {
        let want = sizes
            .as_mut()
            .map_or(input.len(), |s| s.next().unwrap())
            .min(input.len() - pos);
        pending.extend_from_slice(&input[pos..pos + want]);
        pos += want;
        loop {
            let p = block.process(&pending, &mut room).unwrap();
            out.extend_from_slice(&room[..p.produced]);
            pending.drain(..p.consumed);
            if (p.consumed == 0 && p.produced == 0) || pending.is_empty() && p.produced < room.len()
            {
                break;
            }
        }
    }
    out
}

fn chunk_plans(seed: u64) -> impl Iterator<Item = ChunkPlan> {
    [ChunkPlan::new(seed, 7), ChunkPlan::new(seed + 1, 333)].into_iter()
}

/// Runs a fresh block over `input` whole and chunked; checks the golden file and the invariance.
fn check_stream<B: Block>(case: &GoldenCase, input: &[B::In], make: impl Fn() -> B)
where
    B::Out: Default + Sample,
{
    let expected = case.array("expected").unwrap();
    let whole = run_block(&mut make(), input, None);
    golden::assert_matches(case, expected, &whole).unwrap();
    for plan in chunk_plans(case.seed) {
        let chunked = run_block(&mut make(), input, Some(plan));
        assert_eq!(chunked.len(), whole.len(), "{}: output count", case.case);
        assert_eq!(
            bit_exact(&whole, &chunked),
            None,
            "{}: chunking changed the output",
            case.case
        );
    }
}

fn usize_param(case: &GoldenCase, name: &str) -> usize {
    usize::try_from(case.param_u64(name).unwrap()).unwrap()
}

// ---------------------------------------------------------------------------------------------
// blocks

#[test]
fn blocks_cases_are_present() {
    assert_eq!(
        golden::list("blocks").unwrap(),
        [
            "am_demod_c32",
            "am_demod_c32_dec4_gap",
            "am_demod_cq15",
            "am_demod_cq15_dec4_gap",
            "ddc_c32_fs2m4_f600k_d10",
            "fm_demod_c32",
            "resampler_rational_c32_48k_to_44k1",
            "resampler_tuned_c32_ppm",
            "resampler_tuned_cq15_ppm",
            "spectrum_c32_1024_hann_avg4",
            "spectrum_cq15_1024_bfp",
            "ssb_demod_lsb_c32_dec2",
            "ssb_demod_lsb_cq15",
            "ssb_demod_usb_c32",
            "ssb_demod_usb_cq15_dec4_loud"
        ]
    );
}

#[test]
fn ddc_block_matches_python_chain() {
    let case = golden::load("blocks", "ddc_c32_fs2m4_f600k_d10").unwrap();
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!()
    };
    let cfg = DdcConfig {
        frequency_hz: case.param_f64("frequency_hz").unwrap(),
        decimation: usize_param(&case, "decimation"),
        ..DdcConfig::default()
    };
    let rate = Rate::hz(case.param_f64("rate_hz").unwrap() as u64).unwrap();
    // The defaults of the block are the parameters of the case.
    let block = DdcBlock::<C32, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap();
    assert_eq!(block.num_taps(), usize_param(&case, "taps"));
    assert_eq!(
        block.param("cutoff_hz").unwrap(),
        sdrk_blocks::ParamValue::F64(case.param_f64("cutoff_hz").unwrap())
    );
    check_stream(&case, input, || {
        DdcBlock::<C32, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap()
    });
}

#[test]
fn fm_demod_block_matches_python_chain() {
    let case = golden::load("blocks", "fm_demod_c32").unwrap();
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!()
    };
    let cfg = FmDemodConfig {
        deviation_hz: case.param_f64("deviation_hz").unwrap(),
        deemphasis_us: case.param_f64("deemphasis_us").unwrap(),
        audio_cutoff_hz: case.param_f64("audio_cutoff_hz").unwrap(),
        audio_decimation: usize_param(&case, "audio_decimation"),
    };
    let rate = Rate::hz(case.param_f64("rate_hz").unwrap() as u64).unwrap();
    check_stream(&case, input, || {
        FmDemodBlock::<C32, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap()
    });
}

fn am_demod_config(case: &GoldenCase) -> (AmDemodConfig, Rate) {
    let cfg = AmDemodConfig {
        dc_cutoff_hz: case.param_f64("dc_cutoff_hz").unwrap(),
        audio_cutoff_hz: case.param_f64("audio_cutoff_hz").unwrap(),
        audio_decimation: usize_param(case, "audio_decimation"),
    };
    let rate = Rate::hz(case.param_f64("rate_hz").unwrap() as u64).unwrap();
    (cfg, rate)
}

#[test]
fn am_demod_block_matches_python_chain_c32() {
    let case = golden::load("blocks", "am_demod_c32").unwrap();
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!()
    };
    let (cfg, rate) = am_demod_config(&case);
    check_stream(&case, input, || {
        AmDemodBlock::<C32, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap()
    });
}

#[test]
fn am_demod_block_matches_python_chain_cq15() {
    let case = golden::load("blocks", "am_demod_cq15").unwrap();
    let Array::CI16(input) = case.array("input").unwrap() else {
        panic!()
    };
    let (cfg, rate) = am_demod_config(&case);
    check_stream(&case, input, || {
        AmDemodBlock::<CQ15, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap()
    });
}

/// The decimated cases carry a stretch of exactly zero input (a closed squelch): the audio
/// filter path and the silence rule of the block are part of the golden chain.
#[test]
fn am_demod_block_with_audio_filter_matches_python_chain_c32() {
    let case = golden::load("blocks", "am_demod_c32_dec4_gap").unwrap();
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!()
    };
    let (cfg, rate) = am_demod_config(&case);
    assert_eq!(cfg.audio_decimation, 4);
    assert_eq!(cfg.audio_taps(rate).unwrap(), usize_param(&case, "taps"));
    check_stream(&case, input, || {
        AmDemodBlock::<C32, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap()
    });
}

#[test]
fn am_demod_block_with_audio_filter_matches_python_chain_cq15() {
    let case = golden::load("blocks", "am_demod_cq15_dec4_gap").unwrap();
    let Array::CI16(input) = case.array("input").unwrap() else {
        panic!()
    };
    let (cfg, rate) = am_demod_config(&case);
    assert_eq!(cfg.audio_decimation, 4);
    assert_eq!(cfg.audio_taps(rate).unwrap(), usize_param(&case, "taps"));
    check_stream(&case, input, || {
        AmDemodBlock::<CQ15, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap()
    });
}

fn ssb_demod_config(case: &GoldenCase) -> (SsbDemodConfig, Rate) {
    let sideband = match case.param_str("sideband").unwrap() {
        "usb" => SsbSideband::Usb,
        "lsb" => SsbSideband::Lsb,
        other => panic!("unknown sideband {other}"),
    };
    let cfg = SsbDemodConfig {
        sideband,
        low_hz: case.param_f64("low_hz").unwrap(),
        high_hz: case.param_f64("high_hz").unwrap(),
        transition_hz: case.param_f64("transition_hz").unwrap(),
        attenuation_db: case.param_f64("attenuation_db").unwrap(),
        audio_decimation: usize_param(case, "audio_decimation"),
        ..SsbDemodConfig::default()
    };
    let rate = Rate::hz(case.param_f64("rate_hz").unwrap() as u64).unwrap();
    assert_eq!(cfg.num_taps(rate).unwrap(), usize_param(case, "taps"));
    (cfg, rate)
}

#[test]
fn ssb_demod_block_matches_python_chain_usb_c32() {
    let case = golden::load("blocks", "ssb_demod_usb_c32").unwrap();
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!()
    };
    let (cfg, rate) = ssb_demod_config(&case);
    assert_eq!(cfg.sideband, SsbSideband::Usb);
    check_stream(&case, input, || {
        SsbDemodBlock::<C32, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap()
    });
}

#[test]
fn ssb_demod_block_matches_python_chain_lsb_c32_decimated() {
    let case = golden::load("blocks", "ssb_demod_lsb_c32_dec2").unwrap();
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!()
    };
    let (cfg, rate) = ssb_demod_config(&case);
    assert_eq!((cfg.sideband, cfg.audio_decimation), (SsbSideband::Lsb, 2));
    check_stream(&case, input, || {
        SsbDemodBlock::<C32, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap()
    });
}

#[test]
fn ssb_demod_block_matches_python_chain_lsb_cq15() {
    let case = golden::load("blocks", "ssb_demod_lsb_cq15").unwrap();
    let Array::CI16(input) = case.array("input").unwrap() else {
        panic!()
    };
    let (cfg, rate) = ssb_demod_config(&case);
    assert_eq!(cfg.sideband, SsbSideband::Lsb);
    check_stream(&case, input, || {
        SsbDemodBlock::<CQ15, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap()
    });
}

#[test]
fn ssb_demod_block_matches_python_chain_usb_cq15_decimated_loud() {
    let case = golden::load("blocks", "ssb_demod_usb_cq15_dec4_loud").unwrap();
    let Array::CI16(input) = case.array("input").unwrap() else {
        panic!()
    };
    let (cfg, rate) = ssb_demod_config(&case);
    assert_eq!((cfg.sideband, cfg.audio_decimation), (SsbSideband::Usb, 4));
    // The case is built to saturate: the expected audio reaches both rails.
    let Array::I16(expected) = case.array("expected").unwrap() else {
        panic!()
    };
    assert!(expected.contains(&i16::MAX) && expected.contains(&i16::MIN));
    check_stream(&case, input, || {
        SsbDemodBlock::<CQ15, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap()
    });
}

#[test]
fn rational_resampler_matches_upfirdn() {
    let case = golden::load("blocks", "resampler_rational_c32_48k_to_44k1").unwrap();
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!()
    };
    let cfg = ResamplerConfig {
        rate_out_hz: case.param_f64("rate_out_hz").unwrap(),
        max_taps: usize_param(&case, "max_taps"),
        ..ResamplerConfig::default()
    };
    let rate = Rate::hz(case.param_f64("rate_in_hz").unwrap() as u64).unwrap();
    let block = ResamplerBlock::<C32, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap();
    assert_eq!(
        block.timing().ratio,
        sdrk::Ratio::new(
            case.param_u64("up").unwrap(),
            case.param_u64("down").unwrap()
        )
        .unwrap()
    );
    check_stream(&case, input, || {
        ResamplerBlock::<C32, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap()
    });
}

fn tuned_config(case: &GoldenCase) -> (ResamplerConfig, Rate) {
    let cfg = ResamplerConfig {
        rate_out_hz: case.param_f64("rate_out_hz").unwrap(),
        tunable: true,
        ppm: case.param_f64("ppm").unwrap(),
        ..ResamplerConfig::default()
    };
    (
        cfg,
        Rate::hz(case.param_f64("rate_in_hz").unwrap() as u64).unwrap(),
    )
}

#[test]
fn tuned_resampler_matches_python_model_c32() {
    let case = golden::load("blocks", "resampler_tuned_c32_ppm").unwrap();
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!()
    };
    let (cfg, rate) = tuned_config(&case);
    check_stream(&case, input, || {
        ResamplerBlock::<C32, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap()
    });
}

#[test]
fn tuned_resampler_matches_python_model_cq15() {
    let case = golden::load("blocks", "resampler_tuned_cq15_ppm").unwrap();
    let Array::CI16(input) = case.array("input").unwrap() else {
        panic!()
    };
    let (cfg, rate) = tuned_config(&case);
    check_stream(&case, input, || {
        ResamplerBlock::<CQ15, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap()
    });
}

fn spectrum_config(case: &GoldenCase) -> SpectrumConfig {
    assert_eq!(case.param_str("window").unwrap(), "hann");
    SpectrumConfig {
        fft_size: usize_param(case, "fft_size"),
        window: SpectrumWindow::Hann,
        averaging: usize_param(case, "averaging"),
        interval_samples: Some(usize_param(case, "interval_samples")),
        ..SpectrumConfig::default()
    }
}

/// Pushes `input` in pieces of the plan and collects every completed frame.
fn frames<B: FrameBlock>(
    block: &mut B,
    input: &[B::In],
    plan: Option<ChunkPlan>,
) -> Vec<(Vec<f32>, sdrk_blocks::FrameMeta)> {
    let mut out = Vec::new();
    let mut sizes = plan.map(|p| p.sizes());
    let mut pos = 0;
    let mut frame = vec![0.0f32; block.frame_len()];
    while pos < input.len() {
        let want = sizes
            .as_mut()
            .map_or(input.len(), |s| s.next().unwrap())
            .min(input.len() - pos);
        let mut piece = &input[pos..pos + want];
        pos += want;
        while !piece.is_empty() {
            let (used, meta) = block.push(piece, &mut frame).unwrap();
            if let Some(meta) = meta {
                out.push((frame.clone(), meta));
            }
            assert!(used > 0, "push made no progress");
            piece = &piece[used..];
        }
    }
    out
}

fn check_spectrum<T: sdrk_blocks::SpectrumSample>(case: &GoldenCase, input: &[T])
where
    Sdrk: sdrk_core::kernels::spectrum::Window<T>
        + sdrk_core::kernels::fft::Fft<T>
        + sdrk_core::kernels::spectrum::PowerDb<f32>,
{
    let cfg = spectrum_config(case);
    let rate = Rate::hz(1_000_000).unwrap();
    let make = || SpectrumBlock::<T, VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap();
    let whole = frames(&mut make(), input, None);
    assert_eq!(whole.len(), 1);
    assert_eq!(whole[0].1.averaged, 4);
    assert_eq!(whole[0].1.first_sample, 0);
    assert_eq!(whole[0].1.exponent, 0);
    golden::assert_matches(case, case.array("expected").unwrap(), &whole[0].0).unwrap();
    for plan in chunk_plans(case.seed) {
        let chunked = frames(&mut make(), input, Some(plan));
        assert_eq!(chunked, whole, "chunking changed the frame");
    }
}

#[test]
fn spectrum_block_c32_matches_numpy() {
    let case = golden::load("blocks", "spectrum_c32_1024_hann_avg4").unwrap();
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!()
    };
    check_spectrum::<C32>(&case, input);
}

#[test]
fn spectrum_block_cq15_matches_numpy() {
    let case = golden::load("blocks", "spectrum_cq15_1024_bfp").unwrap();
    let Array::CI16(input) = case.array("input").unwrap() else {
        panic!()
    };
    check_spectrum::<CQ15>(&case, input);
}
