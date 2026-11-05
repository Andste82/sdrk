//! Block equals kernel (M32): each block computes exactly what its kernel computes with the
//! parameters the block derives from its configuration, in one piece and over 20 cut plans of
//! seed 470.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::large_stack_arrays,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::multiple_bound_locations,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::float_cmp,
    clippy::cast_possible_truncation
)]

mod common;

use common::{
    check_block, check_block_with, rate, real_signal, run_block, same_bits, signal, twenty_plans,
};
use num_complex::Complex64;
use sdrk::{C32, CQ15, Q15, RealSample, Sample, SampleType, Sdrk};
use sdrk_blocks::design::{
    BiquadResponse, biquad_coefs, butterworth_sos, halfband_len, halfband_taps, kaiser_beta,
    to_f32_sos, to_q14,
};
use sdrk_blocks::{
    AutoNotchConfig, AutoNotchRealBlock, BiquadBlock, BiquadConfig, BiquadElem, BiquadRealBlock,
    Block, BlockReal, BlockSample, ConvertBlock, ConvertConfig, ConvertOutBlock, ConvertOutConfig,
    DcBlockConfig, DcBlockRealBlock, DesignSpec, FirBlock, FirConfig, FmModBlock, FmModConfig,
    HalfbandDecimateConfig, NoiseBlankBlock, NoiseBlankConfig, ParamValue, RealToIqBlock,
    StreamElem,
};
use sdrk_core::kernels::auto_notch::{AutoNotch, AutoNotchState};
use sdrk_core::kernels::biquad::{Biquad, BiquadReal, BiquadRealState, BiquadState};
use sdrk_core::kernels::convert::{Convert, ConvertOut};
use sdrk_core::kernels::demod::{DcBlockReal, DcBlockRealState};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::halfband::{RealToComplex, RealToComplexState};
use sdrk_core::kernels::modulate::{FmMod, FmModState};
use sdrk_core::kernels::noise_blank::{NoiseBlank, NoiseBlankState};
use sdrk_device::{SampleFormat, formats};
use sdrk_testkit::{ChunkPlan, Pcg32, VecStorage};

/// Seed of the cut plans of M32 (seed table of docs/milestones/README.md).
const SEED: u64 = 470;

// ---- dc_block (real) --------------------------------------------------------------------

/// `DcBlockRealBlock` equals `dc_block_real` with the pole `r = exp(-2 pi cutoff_hz / fs)`,
/// rounded once to the sample type. The `f32` kernel of the SIMD backend is a tolerance kernel
/// whose bits may depend on the split (docs/kernels.md), so its cut plans are checked at 120 dB.
fn dc_block_real_equals_kernel<R: BlockReal>(cutoff_hz: f64, fs: u64, offset: f64)
where
    Sdrk: DcBlockReal<R>,
    R: PartialEq + core::fmt::Debug,
{
    let cfg = DcBlockConfig { cutoff_hz };
    // A tone with noise on a DC offset, within full scale.
    let mut rng = Pcg32::new(SEED, 11);
    let input: Vec<R> = (0..4000)
        .map(|i| {
            let tone = 0.3 * (std::f64::consts::TAU * 0.01 * i as f64).sin();
            R::from_f64_round(offset + tone + 0.02 * rng.normal())
        })
        .collect();
    // The documented pole, computed with `libm` as the block computes it.
    let r = libm::exp(-2.0 * std::f64::consts::PI * cutoff_hz / fs as f64);
    let mut state = DcBlockRealState::<R>::new(R::from_f64_round(r)).unwrap();
    let mut want = vec![R::ZERO; input.len()];
    <Sdrk as DcBlockReal<R>>::dc_block_real(&Sdrk, &mut state, &input, &mut want).unwrap();
    check_block_with(
        &want,
        &input,
        SEED,
        || DcBlockRealBlock::<R, _>::new(&cfg, rate(fs), &mut VecStorage).unwrap(),
        &format!("dc_block_real {cutoff_hz} Hz at {fs} Hz"),
        R::TYPE != sdrk::SampleType::F32,
    );
}

#[test]
fn dc_block_real_equals_the_kernel() {
    // The default (10 Hz at 48 kHz) and the corner of examples/flows/spectrum_only.json
    // (100 Hz at 2.4 MHz).
    for (cutoff_hz, fs, offset) in [(10.0, 48_000, 0.3), (100.0, 2_400_000, -0.4)] {
        dc_block_real_equals_kernel::<f32>(cutoff_hz, fs, offset);
        dc_block_real_equals_kernel::<Q15>(cutoff_hz, fs, offset);
    }
    // Extreme but valid: a corner just below half the sample rate (a pole near exp(-pi)).
    dc_block_real_equals_kernel::<f32>(23_999.0, 48_000, 0.1);
    dc_block_real_equals_kernel::<Q15>(23_999.0, 48_000, 0.1);
}

// ---- shared helpers of the runtime parameter changes ------------------------------------

/// Runs a fresh block over `input[..at]`, applies `change`, runs it over `input[at..]` (both
/// parts in pieces from `plan`, all at once for `None`) and returns the whole output.
fn run_with_change<B, F, G>(
    make: &F,
    change: &G,
    input: &[B::In],
    at: usize,
    plan: Option<ChunkPlan>,
) -> Vec<B::Out>
where
    B: Block,
    B::In: StreamElem,
    B::Out: StreamElem,
    F: Fn() -> B,
    G: Fn(&mut B),
{
    let mut block = make();
    let mut out = run_block(&mut block, &input[..at], plan);
    change(&mut block);
    out.extend(run_block(&mut block, &input[at..], plan));
    out
}

/// The block with a parameter change at sample `at` equals `want`, in one piece and over the
/// 20 cut plans of seed 470 (bit for bit: only bit-exact kernels have runtime parameters).
fn check_change<B, F, G>(
    want: &[B::Out],
    input: &[B::In],
    at: usize,
    make: F,
    change: G,
    what: &str,
) where
    B: Block,
    B::In: StreamElem,
    B::Out: StreamElem + PartialEq + core::fmt::Debug,
    F: Fn() -> B,
    G: Fn(&mut B),
{
    let whole = run_with_change(&make, &change, input, at, None);
    same_bits(want, &whole, &format!("{what} whole"));
    for plan in twenty_plans(SEED) {
        let chunked = run_with_change(&make, &change, input, at, Some(plan));
        same_bits(want, &chunked, &format!("{what} {plan:?}"));
    }
}

/// Whether the kernel of a `C32` block is a tolerance kernel of the SIMD backend whose bits may
/// depend on the split (docs/kernels.md): its cut plans are then checked at 120 dB.
fn exact_chunks<T: Sample>(tolerance_for_c32: bool) -> bool {
    !(tolerance_for_c32 && T::TYPE == SampleType::C32)
}

// ---- fir --------------------------------------------------------------------------------

/// A designed tap quantized by the rounding rule R7 of `sdrk-core` (`RealSample::from_f64_round`),
/// with the range check of `design::to_q15`. The test does not use the blocks' own
/// `TunedElem::tap`, so a wrong quantization in the blocks shows up as a difference.
fn r7_tap<R: RealSample>(v: f64) -> R {
    assert!(
        v.abs() <= (1.0 - 1.0 / 32768.0) * 1.01,
        "tap {v} out of range"
    );
    R::from_f64_round(v)
}

/// `FirBlock` equals `fir` with the taps of `DesignSpec::design` (Kaiser window, `kaiserord`)
/// quantized by the rounding rule R7 (`r7_tap`; `CQ15`: the Q15 rounding of `design::to_q15`).
/// The `C32` `fir` is a tolerance kernel of the SIMD backend, so its cut plans are checked at
/// 120 dB; `CQ15` is bit-exact everywhere.
fn fir_equals_kernel<T: BlockSample>(cfg: &FirConfig, fs: u64)
where
    Sdrk: Fir<T>,
    T: PartialEq + core::fmt::Debug,
{
    let spec = DesignSpec {
        cutoff_hz: cfg.cutoff_hz,
        transition_hz: cfg.transition_hz,
        attenuation_db: cfg.attenuation_db,
    };
    let mut design = vec![0.0f64; spec.num_taps(fs as f64, 1023).unwrap()];
    spec.design(fs as f64, 1023, &mut design).unwrap();
    let taps: Vec<T::Real> = design.iter().map(|v| r7_tap(*v)).collect();
    let n = taps.len();
    let block = FirBlock::<T, _>::new(cfg, rate(fs), &mut VecStorage).unwrap();
    assert_eq!(block.num_taps(), n, "fir {cfg:?}: tap count");
    let mut input = signal::<T>(4000, SEED, 0.03);
    // Full-scale stretches drive the CQ15 sums to their saturation limits.
    for x in &mut input[1000..1040] {
        *x = T::from_c64_round(Complex64::new(1.0, -1.0));
    }
    for x in &mut input[2500..2540] {
        *x = T::from_c64_round(Complex64::new(-1.0, 1.0));
    }
    let mut state = FirState::<T, _, _>::new(taps, vec![T::ZERO; n - 1]).unwrap();
    let mut want = vec![T::ZERO; input.len()];
    <Sdrk as Fir<T>>::fir(&Sdrk, &mut state, &input, &mut want).unwrap();
    check_block_with(
        &want,
        &input,
        SEED,
        || FirBlock::<T, _>::new(cfg, rate(fs), &mut VecStorage).unwrap(),
        &format!("fir {cfg:?} at {fs} Hz ({n} taps)"),
        exact_chunks::<T>(true),
    );
}

#[test]
fn fir_equals_the_kernel() {
    let configs = [
        // The defaults.
        FirConfig::default(),
        // Extreme: the stop band starts exactly at half the rate, 120 dB.
        FirConfig {
            cutoff_hz: 23_000.0,
            transition_hz: 2_000.0,
            attenuation_db: 120.0,
            ..FirConfig::default()
        },
        // Extreme: the smallest attenuation (a Kaiser window with beta 0).
        FirConfig {
            cutoff_hz: 1_000.0,
            transition_hz: 1_500.0,
            attenuation_db: 8.0,
            ..FirConfig::default()
        },
        // Extreme: a narrow transition that needs several hundred taps.
        FirConfig {
            cutoff_hz: 3_000.0,
            transition_hz: 200.0,
            ..FirConfig::default()
        },
    ];
    for cfg in &configs {
        fir_equals_kernel::<C32>(cfg, 48_000);
        fir_equals_kernel::<CQ15>(cfg, 48_000);
    }
}

// ---- biquad -----------------------------------------------------------------------------

/// The sections the block must design for `cfg` at `fs`, written out with the public design
/// functions the block documents: `butterworth_sos` for a low-pass or high-pass, `biquad_coefs`
/// for a band-pass or notch, then `to_q14` (Q15 types) or `to_f32_sos`.
fn biquad_sections<T: BiquadElem>(cfg: &BiquadConfig, fs: f64) -> Vec<[T::Coef; 5]> {
    let mut sos = vec![[0.0f64; 5]; cfg.sections()];
    match cfg.response {
        BiquadResponse::Lowpass | BiquadResponse::Highpass => {
            butterworth_sos(cfg.response, cfg.order, cfg.cutoff_hz, fs, &mut sos).unwrap();
        }
        _ => sos[0] = biquad_coefs(cfg.response, cfg.cutoff_hz, cfg.q, fs).unwrap(),
    }
    if T::Q15 {
        let mut q = vec![[0i16; 5]; sos.len()];
        to_q14(&sos, &mut q).unwrap();
        bytemuck::cast_slice::<[i16; 5], [T::Coef; 5]>(&q).to_vec()
    } else {
        let mut f = vec![[0.0f32; 5]; sos.len()];
        to_f32_sos(&sos, &mut f).unwrap();
        bytemuck::cast_slice::<[f32; 5], [T::Coef; 5]>(&f).to_vec()
    }
}

/// The configuration after the runtime change of the test: the cutoff moves by a quarter
/// (up, or down where up would leave the valid range) and `q` doubles.
fn biquad_changed(cfg: &BiquadConfig, fs: f64) -> BiquadConfig {
    let up = cfg.cutoff_hz * 1.25;
    BiquadConfig {
        cutoff_hz: if up < fs / 2.0 {
            up
        } else {
            cfg.cutoff_hz * 0.8
        },
        q: cfg.q * 2.0,
        ..*cfg
    }
}

/// Sets the runtime parameters of `next` on a biquad block.
fn biquad_set<B: Block>(block: &mut B, next: &BiquadConfig) {
    block
        .set_param("cutoff_hz", ParamValue::F64(next.cutoff_hz))
        .unwrap();
    block.set_param("q", ParamValue::F64(next.q)).unwrap();
}

/// Sample at which the runtime change of the tests happens (not on a boundary of 4 or 64).
const CHANGE_AT: usize = 1789;

/// `BiquadBlock` equals `biquad` with the designed sections, also across a runtime change of
/// `cutoff_hz` and `q` (the state is kept, only the coefficients change).
fn biquad_equals_kernel<T>(cfg: &BiquadConfig, fs: u64)
where
    T: BlockSample + BiquadElem + PartialEq + core::fmt::Debug,
    Sdrk: Biquad<T>,
{
    let f = fs as f64;
    let input = signal::<T>(4000, SEED, 0.02);
    let n = cfg.sections();
    let make = || BiquadBlock::<T, _>::new(cfg, rate(fs), &mut VecStorage).unwrap();
    let what = format!("biquad {:?} {cfg:?} at {fs} Hz", T::TYPE);
    let mut state =
        BiquadState::<T, _, _>::new(biquad_sections::<T>(cfg, f), vec![T::ZERO; 4 * n]).unwrap();
    let mut want = vec![T::ZERO; input.len()];
    <Sdrk as Biquad<T>>::biquad(&Sdrk, &mut state, &input, &mut want).unwrap();
    check_block(&want, &input, SEED, make, &what);

    let next = biquad_changed(cfg, f);
    let mut state =
        BiquadState::<T, _, _>::new(biquad_sections::<T>(cfg, f), vec![T::ZERO; 4 * n]).unwrap();
    let mut want = vec![T::ZERO; input.len()];
    let (a, b) = want.split_at_mut(CHANGE_AT);
    <Sdrk as Biquad<T>>::biquad(&Sdrk, &mut state, &input[..CHANGE_AT], a).unwrap();
    state
        .coefs_mut()
        .copy_from_slice(&biquad_sections::<T>(&next, f));
    <Sdrk as Biquad<T>>::biquad(&Sdrk, &mut state, &input[CHANGE_AT..], b).unwrap();
    check_change(
        &want,
        &input,
        CHANGE_AT,
        make,
        |blk| biquad_set(blk, &next),
        &format!("{what} changed to {next:?}"),
    );
}

/// `BiquadRealBlock` equals `biquad_real`, as [`biquad_equals_kernel`].
fn biquad_real_equals_kernel<R>(cfg: &BiquadConfig, fs: u64)
where
    R: BlockReal + BiquadElem + PartialEq + core::fmt::Debug,
    Sdrk: BiquadReal<R>,
{
    let f = fs as f64;
    let input = real_signal::<R>(4000, SEED, 0.02);
    let n = cfg.sections();
    let make = || BiquadRealBlock::<R, _>::new(cfg, rate(fs), &mut VecStorage).unwrap();
    let what = format!("biquad {:?} {cfg:?} at {fs} Hz", R::TYPE);
    let mut state =
        BiquadRealState::<R, _, _>::new(biquad_sections::<R>(cfg, f), vec![R::ZERO; 4 * n])
            .unwrap();
    let mut want = vec![R::ZERO; input.len()];
    <Sdrk as BiquadReal<R>>::biquad_real(&Sdrk, &mut state, &input, &mut want).unwrap();
    check_block(&want, &input, SEED, make, &what);

    let next = biquad_changed(cfg, f);
    let mut state =
        BiquadRealState::<R, _, _>::new(biquad_sections::<R>(cfg, f), vec![R::ZERO; 4 * n])
            .unwrap();
    let mut want = vec![R::ZERO; input.len()];
    let (a, b) = want.split_at_mut(CHANGE_AT);
    <Sdrk as BiquadReal<R>>::biquad_real(&Sdrk, &mut state, &input[..CHANGE_AT], a).unwrap();
    state
        .coefs_mut()
        .copy_from_slice(&biquad_sections::<R>(&next, f));
    <Sdrk as BiquadReal<R>>::biquad_real(&Sdrk, &mut state, &input[CHANGE_AT..], b).unwrap();
    check_change(
        &want,
        &input,
        CHANGE_AT,
        make,
        |blk| biquad_set(blk, &next),
        &format!("{what} changed to {next:?}"),
    );
}

/// Every response and order the block supports, at 48 kHz: the defaults, the filters of the
/// example flows (`am_receiver` 4 kHz low-pass, `ssb_receiver` 2.8 kHz low-pass, `nfm_receiver`
/// 300 Hz high-pass, all of order 4) and extremes (the Q15 cutoff limit fs / 500, a corner near
/// half the rate, a very narrow notch, a very wide band-pass).
fn biquad_configs() -> Vec<BiquadConfig> {
    let mk = |response, order, cutoff_hz, q| BiquadConfig {
        response,
        order,
        cutoff_hz,
        q,
    };
    let d = BiquadConfig::default();
    let mut v = vec![d];
    for order in [2, 4, 6, 8] {
        v.push(mk(BiquadResponse::Lowpass, order, 4_000.0, d.q));
        v.push(mk(BiquadResponse::Highpass, order, 300.0, d.q));
    }
    v.extend([
        mk(BiquadResponse::Lowpass, 4, 2_800.0, d.q),
        mk(BiquadResponse::Lowpass, 8, 96.0, d.q),
        mk(BiquadResponse::Highpass, 8, 96.0, d.q),
        mk(BiquadResponse::Lowpass, 2, 23_000.0, d.q),
        mk(BiquadResponse::Highpass, 8, 22_000.0, d.q),
        mk(BiquadResponse::Bandpass, 2, 3_000.0, 2.0),
        mk(BiquadResponse::Bandpass, 2, 1_000.0, 0.05),
        mk(BiquadResponse::Notch, 2, 1_000.0, 10.0),
        mk(BiquadResponse::Notch, 2, 18_000.0, 400.0),
    ]);
    v
}

#[test]
fn biquad_equals_the_kernel() {
    for cfg in &biquad_configs() {
        biquad_equals_kernel::<C32>(cfg, 48_000);
        biquad_equals_kernel::<CQ15>(cfg, 48_000);
    }
}

#[test]
fn biquad_real_equals_the_kernel() {
    for cfg in &biquad_configs() {
        biquad_real_equals_kernel::<f32>(cfg, 48_000);
        biquad_real_equals_kernel::<Q15>(cfg, 48_000);
    }
}

// ---- real_to_iq -------------------------------------------------------------------------

/// `RealToIqBlock` equals `real_to_complex` with the halfband taps of `halfband_taps`
/// (`halfband_len` taps, Kaiser beta from `kaiser_beta`, the transition defaulting to 0.1 fs),
/// quantized by the rounding rule R7 (`r7_tap`). The `C32` kernel is a tolerance kernel of the SIMD backend
/// (cut plans at 120 dB); `CQ15` is bit-exact everywhere.
fn real_to_iq_equals_kernel<T: BlockSample>(cfg: &HalfbandDecimateConfig, fs: u64)
where
    Sdrk: RealToComplex<T>,
    T: PartialEq + core::fmt::Debug,
{
    let f = fs as f64;
    let transition = cfg.transition_hz.unwrap_or(0.1 * f);
    let n = halfband_len(transition, cfg.attenuation_db, f).unwrap();
    assert!(n <= cfg.max_taps);
    let mut design = vec![0.0f64; n];
    halfband_taps(kaiser_beta(cfg.attenuation_db), &mut design).unwrap();
    let taps: Vec<T::Real> = design.iter().map(|v| r7_tap(*v)).collect();
    let block = RealToIqBlock::<T, _>::new(cfg, rate(fs), &mut VecStorage).unwrap();
    assert_eq!(block.num_taps(), n, "real_to_iq {cfg:?}: tap count");

    let mut input = real_signal::<T::Real>(4000, SEED, 0.17);
    // Full-scale runs (CQ15: -32768 at every phase of n mod 4, the saturating negation).
    for x in &mut input[1001..1043] {
        *x = <T::Real as RealSample>::from_f64_round(-1.0);
    }
    for x in &mut input[2502..2541] {
        *x = <T::Real as RealSample>::from_f64_round(1.0);
    }
    let mut state = RealToComplexState::<T, _, _>::new(taps, vec![T::ZERO; n - 1]).unwrap();
    let mut want = vec![T::ZERO; input.len() / 2];
    let p =
        <Sdrk as RealToComplex<T>>::real_to_complex(&Sdrk, &mut state, &input, &mut want).unwrap();
    assert_eq!((p.consumed, p.produced), (input.len(), want.len()));
    check_block_with(
        &want,
        &input,
        SEED,
        || RealToIqBlock::<T, _>::new(cfg, rate(fs), &mut VecStorage).unwrap(),
        &format!("real_to_iq {:?} {cfg:?} at {fs} Hz ({n} taps)", T::TYPE),
        exact_chunks::<T>(true),
    );
}

#[test]
fn real_to_iq_equals_the_kernel() {
    let d = HalfbandDecimateConfig::default();
    let cases = [
        // The defaults.
        (d, 48_000),
        // examples/flows/real_input_fm_receiver.json: rs16 at 4.8 MHz.
        (
            HalfbandDecimateConfig {
                transition_hz: Some(480_000.0),
                attenuation_db: 60.0,
                ..d
            },
            4_800_000,
        ),
        // Extreme: a narrow transition at 100 dB (several hundred taps).
        (
            HalfbandDecimateConfig {
                transition_hz: Some(960.0),
                attenuation_db: 100.0,
                ..d
            },
            48_000,
        ),
        // Extreme: the smallest attenuation and a transition close to half the rate.
        (
            HalfbandDecimateConfig {
                transition_hz: Some(21_600.0),
                attenuation_db: 8.0,
                ..d
            },
            48_000,
        ),
    ];
    for (cfg, fs) in &cases {
        real_to_iq_equals_kernel::<C32>(cfg, *fs);
        real_to_iq_equals_kernel::<CQ15>(cfg, *fs);
    }
}

// ---- convert ----------------------------------------------------------------------------

/// The formats of the `convert` and `convert_out` tests: the three common integer formats and
/// both packed formats (PACKED12: one complex sample in 3 bytes; PACKED10: two in 5 bytes).
const FORMATS: [(&str, SampleFormat); 5] = [
    ("cu8", formats::CU8),
    ("cs8", formats::CS8),
    ("cs16", formats::CS16),
    ("packed12", formats::PACKED12),
    ("packed10", formats::PACKED10),
];

/// `ConvertBlock` equals `convert` with the configured format. The bytes are random (every
/// code) and the cut plans of seed 470 split them inside samples and packing groups; the block
/// must keep a partial group for the next call.
fn convert_equals_kernel<T: BlockSample>(name: &str, format: SampleFormat)
where
    Sdrk: Convert<T>,
    T: PartialEq + core::fmt::Debug,
{
    let samples = 2400;
    let bytes = format.bytes_for_samples(samples).unwrap();
    let mut rng = Pcg32::new(SEED, 13);
    let input: Vec<u8> = (0..bytes).map(|_| rng.next_u32() as u8).collect();
    let mut want = vec![T::ZERO; samples];
    let produced = <Sdrk as Convert<T>>::convert(&Sdrk, &format, &input, &mut want).unwrap();
    assert_eq!(produced, samples);
    let cfg = ConvertConfig { format };
    check_block(
        &want,
        &input,
        SEED,
        || ConvertBlock::<T, _>::new(&cfg, rate(2_400_000), &mut VecStorage).unwrap(),
        &format!("convert {name} to {:?}", T::TYPE),
    );
}

#[test]
fn convert_equals_the_kernel() {
    for (name, format) in FORMATS {
        convert_equals_kernel::<C32>(name, format);
        convert_equals_kernel::<CQ15>(name, format);
    }
}

// ---- convert_out ------------------------------------------------------------------------

/// `ConvertOutBlock` equals `convert_out` byte for byte. The input is a tone with noise plus
/// the corner values of the encoding: full scale, beyond full scale, and (`C32`) NaN and
/// infinities.
fn convert_out_equals_kernel<T: BlockSample>(name: &str, format: SampleFormat, corners: &[T])
where
    Sdrk: ConvertOut<T>,
{
    let mut input = signal::<T>(2400, SEED, 0.01);
    input[..corners.len()].copy_from_slice(corners);
    let bytes = format.bytes_for_samples(input.len()).unwrap();
    let mut want = vec![0u8; bytes];
    let written = <Sdrk as ConvertOut<T>>::convert_out(&Sdrk, &format, &input, &mut want).unwrap();
    assert_eq!(written, bytes);
    let cfg = ConvertOutConfig { format };
    check_block(
        &want,
        &input,
        SEED,
        || ConvertOutBlock::<T, _>::new(&cfg, rate(2_400_000), &mut VecStorage).unwrap(),
        &format!("convert_out {:?} to {name}", T::TYPE),
    );
}

#[test]
fn convert_out_equals_the_kernel() {
    let c = C32::new;
    let c32 = [
        c(1.0, -1.0),
        c(1.5, -1.5),
        c(f32::NAN, f32::INFINITY),
        c(f32::NEG_INFINITY, 0.0),
        c(0.5, -0.5),
        c(1.0 / 256.0, -1.0 / 256.0),
        c(0.5 / 32768.0, -0.5 / 32768.0),
        c(1.5 / 128.0, -1.5 / 128.0),
    ];
    let q = |re: i16, im: i16| CQ15 {
        re: Q15(re),
        im: Q15(im),
    };
    let cq15 = [
        q(i16::MAX, i16::MIN),
        q(i16::MIN, i16::MAX),
        q(128, -128),
        q(64, -64),
        q(-1, 1),
        q(0, 0),
    ];
    for (name, format) in FORMATS {
        convert_out_equals_kernel::<C32>(name, format, &c32);
        convert_out_equals_kernel::<CQ15>(name, format, &cq15);
    }
}

// ---- fm_mod -----------------------------------------------------------------------------

/// `FmModBlock` equals `fm_mod` with the state `FmModState::new(deviation_hz, rate)`, whose
/// scale is the documented phase step `k = deviation_hz / fs * 2^32` per unit of amplitude;
/// the phase is carried across buffers. A runtime change of `deviation_hz` keeps the phase:
/// the kernel continues with a new state at the same phase. `fm_mod` is bit-exact with the
/// fallback on every backend, so every cut plan is compared bit for bit.
fn fm_mod_equals_kernel<T: BlockSample>(deviation_hz: f64, fs: u64, next_hz: f64)
where
    Sdrk: FmMod<T>,
    T: PartialEq + core::fmt::Debug,
{
    let f = fs as f64;
    let cfg = FmModConfig { deviation_hz };
    let mut rng = Pcg32::new(SEED, 17);
    let mut input: Vec<T::Real> = (0..4000)
        .map(|i| {
            let tone = 0.9 * (std::f64::consts::TAU * 0.003 * i as f64).sin();
            <T::Real as RealSample>::from_f64_round(tone + 0.05 * rng.normal())
        })
        .collect();
    // Full scale in both directions.
    for x in &mut input[500..540] {
        *x = <T::Real as RealSample>::from_f64_round(1.0);
    }
    for x in &mut input[3000..3040] {
        *x = <T::Real as RealSample>::from_f64_round(-1.0);
    }
    let what = format!("fm_mod {:?} {deviation_hz} Hz at {fs} Hz", T::TYPE);
    let make = || FmModBlock::<T, _>::new(&cfg, rate(fs), &mut VecStorage).unwrap();

    let mut state = FmModState::new(deviation_hz, rate(fs)).unwrap();
    assert_eq!(
        state.scale(),
        deviation_hz / f * 4_294_967_296.0,
        "{what}: scale"
    );
    let mut want = vec![T::ZERO; input.len()];
    <Sdrk as FmMod<T>>::fm_mod(&Sdrk, &mut state, &input, &mut want).unwrap();
    check_block(&want, &input, SEED, make, &what);

    let mut state = FmModState::new(deviation_hz, rate(fs)).unwrap();
    let mut want = vec![T::ZERO; input.len()];
    let (a, b) = want.split_at_mut(CHANGE_AT);
    <Sdrk as FmMod<T>>::fm_mod(&Sdrk, &mut state, &input[..CHANGE_AT], a).unwrap();
    let phase = state.phase();
    let mut state = FmModState::new(next_hz, rate(fs)).unwrap();
    state.set_phase(phase);
    <Sdrk as FmMod<T>>::fm_mod(&Sdrk, &mut state, &input[CHANGE_AT..], b).unwrap();
    check_change(
        &want,
        &input,
        CHANGE_AT,
        make,
        |blk| {
            blk.set_param("deviation_hz", ParamValue::F64(next_hz))
                .unwrap();
        },
        &format!("{what} changed to {next_hz} Hz"),
    );
}

#[test]
fn fm_mod_equals_the_kernel() {
    // The default (75 kHz) at the 240 kHz of examples/flows/fm_transmitter.json, the 5 kHz of
    // nfm_transmitter.json, a deviation that makes the rounding of the step visible (33 kHz at
    // 192 kHz, as the golden cases) and the largest valid deviation (half the rate).
    for (deviation_hz, fs, next_hz) in [
        (75_000.0, 240_000, 5_000.0),
        (5_000.0, 240_000, 75_000.0),
        (33_000.0, 192_000, 33_333.3),
        (24_000.0, 48_000, 0.0),
    ] {
        fm_mod_equals_kernel::<C32>(deviation_hz, fs, next_hz);
        fm_mod_equals_kernel::<CQ15>(deviation_hz, fs, next_hz);
    }
}

// ---- noise_blank (M33) ------------------------------------------------------------------

/// A tone of amplitude 0.05 with noise, 30 impulses of 3 samples at (0.9, -0.9) (28 dB above
/// the tone) and a full-scale stretch of 40 samples, the kind of input the blanker is made for.
fn impulsive<T: BlockSample>(n: usize) -> Vec<T> {
    let mut rng = Pcg32::new(SEED, 23);
    let mut input: Vec<T> = (0..n)
        .map(|i| {
            let phase = std::f64::consts::TAU * 0.02 * i as f64;
            let noise = Complex64::new(rng.normal(), rng.normal()) * 0.005;
            T::from_c64_round(Complex64::new(0.05 * phase.cos(), 0.05 * phase.sin()) + noise)
        })
        .collect();
    for _ in 0..30 {
        let at = 100 + rng.below((n - 200) as u32) as usize;
        for x in &mut input[at..at + 3] {
            *x = T::from_c64_round(Complex64::new(0.9, -0.9));
        }
    }
    for x in &mut input[n / 2..n / 2 + 40] {
        *x = T::from_c64_round(Complex64::new(1.0, -1.0));
    }
    input
}

/// `NoiseBlankBlock` equals `noise_blank` with the documented parameters
/// (`D = floor(lookahead_s fs + 0.5)` samples of history; threshold, time constant and hold as
/// given), in one piece and over the 20 cut plans; a runtime change of `threshold_db` at sample
/// `CHANGE_AT` equals the kernel with `set_threshold_db` at the same sample. The kernel stays
/// with the fallback on every backend, so every comparison is bit for bit.
fn noise_blank_equals_kernel<T: BlockSample>(cfg: &NoiseBlankConfig, fs: u64, next_db: f64)
where
    Sdrk: NoiseBlank<T>,
    T: PartialEq + core::fmt::Debug,
{
    let input = impulsive::<T>(4000);
    let d = (cfg.lookahead_s * fs as f64 + 0.5).floor() as usize;
    let what = format!("noise_blank {:?} {cfg:?} at {fs} Hz", T::TYPE);
    let make = || NoiseBlankBlock::<T, _>::new(cfg, rate(fs), &mut VecStorage).unwrap();
    assert_eq!(make().lookahead(), d, "{what}: lookahead");
    let new_state = || {
        NoiseBlankState::<T, _>::new(
            cfg.threshold_db,
            cfg.tau_s,
            cfg.hold_s,
            vec![T::ZERO; d],
            rate(fs),
        )
        .unwrap()
    };

    let mut state = new_state();
    let mut want = vec![T::ZERO; input.len()];
    <Sdrk as NoiseBlank<T>>::noise_blank(&Sdrk, &mut state, &input, &mut want).unwrap();
    assert!(state.blanked() > 0, "{what}: nothing blanked");
    check_block(&want, &input, SEED, make, &what);
    let mut block = make();
    run_block(&mut block, &input, None);
    assert_eq!(block.blanked(), state.blanked(), "{what}: blanked count");

    let mut state = new_state();
    let mut want = vec![T::ZERO; input.len()];
    let (a, b) = want.split_at_mut(CHANGE_AT);
    <Sdrk as NoiseBlank<T>>::noise_blank(&Sdrk, &mut state, &input[..CHANGE_AT], a).unwrap();
    state.set_threshold_db(next_db).unwrap();
    <Sdrk as NoiseBlank<T>>::noise_blank(&Sdrk, &mut state, &input[CHANGE_AT..], b).unwrap();
    check_change(
        &want,
        &input,
        CHANGE_AT,
        make,
        |blk| {
            blk.set_param("threshold_db", ParamValue::F64(next_db))
                .unwrap();
        },
        &format!("{what} changed to {next_db} dB"),
    );
}

#[test]
fn noise_blank_equals_the_kernel() {
    let configs = [
        // The defaults at the 192 kHz of examples/flows/hf_am_receiver.json (D = 2, H = 0).
        (NoiseBlankConfig::default(), 192_000, 25.0),
        // A lookahead and a hold at 192 kHz (D = 10, H = 38).
        (
            NoiseBlankConfig {
                hold_s: 0.0002,
                lookahead_s: 0.00005,
                ..NoiseBlankConfig::default()
            },
            192_000,
            25.0,
        ),
        // Extreme: no lookahead and no hold.
        (
            NoiseBlankConfig {
                hold_s: 0.0,
                lookahead_s: 0.0,
                ..NoiseBlankConfig::default()
            },
            48_000,
            6.0,
        ),
        // Extreme: the longest lookahead and hold, the shortest time constant, the lowest and
        // the highest threshold.
        (
            NoiseBlankConfig {
                threshold_db: 6.0,
                tau_s: 0.001,
                hold_s: 0.01,
                lookahead_s: 0.001,
            },
            48_000,
            40.0,
        ),
    ];
    for (cfg, fs, next_db) in &configs {
        noise_blank_equals_kernel::<C32>(cfg, *fs, *next_db);
        noise_blank_equals_kernel::<CQ15>(cfg, *fs, *next_db);
    }
}

// ---- auto_notch (M33) -------------------------------------------------------------------

/// `AutoNotchRealBlock` equals `auto_notch` with `taps` zero weights, a history of
/// `taps + delay` samples and `mu` as `f32`, in one piece and over the 20 cut plans; a runtime
/// change of `mu` at sample `CHANGE_AT` equals the kernel with `set_mu` at the same sample.
/// The kernel stays with the fallback, so every comparison is bit for bit.
fn auto_notch_equals_kernel(cfg: &AutoNotchConfig, next_mu: f64) {
    let fs = 8_000;
    // Two tones and noise; a NaN and an infinity in the middle.
    let mut input: Vec<f32> = {
        let mut rng = Pcg32::new(SEED, 29);
        (0..4000)
            .map(|i| {
                let t = i as f64 / fs as f64;
                let v = 0.5 * (std::f64::consts::TAU * 1000.0 * t).sin()
                    + 0.3 * (std::f64::consts::TAU * 1700.0 * t).sin()
                    + 0.1 * rng.normal();
                v as f32
            })
            .collect()
    };
    input[1500] = f32::NAN;
    input[2500] = f32::INFINITY;
    let what = format!("auto_notch {cfg:?}");
    let make = || AutoNotchRealBlock::new(cfg, rate(fs), &mut VecStorage).unwrap();
    let new_state = || {
        AutoNotchState::new(
            vec![0.0f32; cfg.taps],
            vec![0.0f32; cfg.taps + cfg.delay],
            cfg.delay,
            cfg.mu as f32,
        )
        .unwrap()
    };

    let mut state = new_state();
    let mut want = vec![0.0f32; input.len()];
    <Sdrk as AutoNotch>::auto_notch(&Sdrk, &mut state, &input, &mut want).unwrap();
    let whole: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
    let mut block = make();
    let got = run_block(&mut block, &input, None);
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        whole,
        "{what}: bits"
    );
    assert_eq!(block.weights(), state.weights(), "{what}: weights");
    check_block(&want, &input, SEED, make, &what);

    let mut state = new_state();
    let mut want = vec![0.0f32; input.len()];
    let (a, b) = want.split_at_mut(CHANGE_AT);
    <Sdrk as AutoNotch>::auto_notch(&Sdrk, &mut state, &input[..CHANGE_AT], a).unwrap();
    state.set_mu(next_mu as f32).unwrap();
    <Sdrk as AutoNotch>::auto_notch(&Sdrk, &mut state, &input[CHANGE_AT..], b).unwrap();
    check_change(
        &want,
        &input,
        CHANGE_AT,
        make,
        |blk| blk.set_param("mu", ParamValue::F64(next_mu)).unwrap(),
        &format!("{what} changed to mu {next_mu}"),
    );
}

#[test]
fn auto_notch_equals_the_kernel() {
    // The defaults, then the extremes of the ranges.
    auto_notch_equals_kernel(&AutoNotchConfig::default(), 0.05);
    auto_notch_equals_kernel(
        &AutoNotchConfig {
            taps: 8,
            delay: 1,
            mu: 0.1,
        },
        0.0001,
    );
    auto_notch_equals_kernel(
        &AutoNotchConfig {
            taps: 256,
            delay: 64,
            mu: 0.0001,
        },
        0.1,
    );
}
