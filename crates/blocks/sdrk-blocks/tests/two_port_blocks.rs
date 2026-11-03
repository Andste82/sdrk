//! The blocks of M30: `add`, `mul`, `make_complex`, `to_complex`, `real_part`, `imag_part`,
//! `nco`, `stereo_pll` and `stereo_matrix`: equality with the kernels, lockstep of the `Two`
//! inputs, chunk invariance, exact storage, parameters and extreme inputs.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::large_stack_arrays,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::multiple_bound_locations
)]

mod common;

use common::{check_exact_storage, plans, rate, real_signal, run_block, signal};
use num_complex::Complex64;
use sdrk::{C32, CQ15, ComplexSample, Q15, Sdrk};
use sdrk_blocks::design::{BiquadResponse, biquad_coefs};
use sdrk_blocks::{
    AddBlock, AddRealBlock, BiquadElem, Block, BlockReal, BlockSample, DesignSpec, Error,
    FmStereoDemodBlock, FmStereoDemodConfig, ImagPartBlock, MakeComplexBlock, MulBlock,
    MulRealBlock, NcoBlock, NcoConfig, NcoRealBlock, ParamValue, RealPartBlock, StereoMatrixBlock,
    StereoPllBlock, StereoPllConfig, StreamElem, ToComplexBlock, TunedElem, Two,
};
use sdrk_core::kernels::biquad::{Biquad, BiquadReal, BiquadRealState};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate};
use sdrk_core::kernels::demod::{QuadDemod, QuadDemodState};
use sdrk_core::kernels::math::{Add, AddReal, Interleave, Mul, MulReal};
use sdrk_core::kernels::mix::{Nco, Oscillate};
use sdrk_core::kernels::stereo::{StereoMatrix, StereoPll, StereoPllState};
use sdrk_testkit::fm::{fm_multitone_baseband, stereo_mpx_tones};
use sdrk_testkit::{ChunkPlan, VecStorage};

fn bytes<T: StreamElem>(v: &[T]) -> &[u8] {
    bytemuck::cast_slice(v)
}

/// Feeds two inputs of equal length to a `Two` block in pieces from `plan` (all at once for
/// `None`); the block must take and produce every piece completely.
fn run_two<A, B, O, K>(block: &mut K, a: &[A], b: &[B], plan: Option<ChunkPlan>) -> Vec<O>
where
    A: StreamElem,
    B: StreamElem,
    O: StreamElem,
    K: Block<In = Two<A, B>, Out = O>,
{
    let mut out = Vec::new();
    let mut sizes = plan.map(|p| p.sizes());
    let mut pos = 0;
    while pos < a.len() {
        let want = sizes
            .as_mut()
            .map_or(a.len(), |s| s.next().unwrap())
            .min(a.len() - pos);
        let mut room = vec![O::default(); want];
        let p = block
            .process((&a[pos..pos + want], &b[pos..pos + want]), &mut room)
            .unwrap();
        assert_eq!((p.consumed, p.produced), (want, want));
        out.extend_from_slice(&room);
        pos += want;
    }
    out
}

/// A call with unequal input lengths is `LengthMismatch` and writes nothing.
fn assert_lockstep<A, B, O, K>(block: &mut K, a: &[A], b: &[B])
where
    A: StreamElem,
    B: StreamElem,
    O: StreamElem,
    K: Block<In = Two<A, B>, Out = O>,
{
    let mut out = vec![O::default(); a.len()];
    for (x, y) in [(&a[..4], &b[..3]), (&a[..3], &b[..4])] {
        let r = block.process((x, y), &mut out);
        assert!(
            matches!(r, Err(Error::Core(sdrk_core::Error::LengthMismatch { .. }))),
            "{r:?}"
        );
        assert!(
            bytes(&out).iter().all(|b| *b == 0),
            "a rejected call wrote output"
        );
    }
}

fn corner<T: ComplexSample>(re: f64, im: f64) -> T {
    T::from_c64_round(Complex64::new(re, im))
}

// ---------------------------------------------------------------------------------------------
// add, mul

fn complex_pair<T: ComplexSample>(n: usize) -> (Vec<T>, Vec<T>) {
    let mut a = signal::<T>(n, 11, 0.01);
    let mut b = signal::<T>(n, 12, 0.03);
    // Corners: full scale of both signs (saturation, -1 times -1).
    for (i, (x, y)) in [
        ((-1.0, -1.0), (-1.0, -1.0)),
        ((1.0, 1.0), (1.0, 1.0)),
        ((-1.0, 0.0), (0.0, -1.0)),
        ((0.5, -0.5), (-1.0, 1.0)),
    ]
    .into_iter()
    .enumerate()
    {
        a[i] = corner(x.0, x.1);
        b[i] = corner(y.0, y.1);
    }
    (a, b)
}

fn complex_math_equals_kernels<T: BlockSample>()
where
    Sdrk: Add<T> + Mul<T>,
{
    let (a, b) = complex_pair::<T>(1000);
    let mut sum = vec![T::default(); a.len()];
    let mut prod = vec![T::default(); a.len()];
    Sdrk.add(&a, &b, &mut sum).unwrap();
    Sdrk.mul(&a, &b, &mut prod).unwrap();
    for plan in std::iter::once(None).chain(plans(31).into_iter().map(Some)) {
        let mut add = AddBlock::<T, _>::new(rate(48_000), &mut VecStorage).unwrap();
        let mut mul = MulBlock::<T, _>::new(rate(48_000), &mut VecStorage).unwrap();
        assert_eq!(bytes(&run_two(&mut add, &a, &b, plan)), bytes(&sum));
        assert_eq!(bytes(&run_two(&mut mul, &a, &b, plan)), bytes(&prod));
    }
    assert_lockstep(
        &mut AddBlock::<T, _>::new(rate(1), &mut VecStorage).unwrap(),
        &a,
        &b,
    );
    assert_lockstep(
        &mut MulBlock::<T, _>::new(rate(1), &mut VecStorage).unwrap(),
        &a,
        &b,
    );
}

fn real_math_equals_kernels<R: BlockReal>()
where
    Sdrk: AddReal<R> + MulReal<R>,
{
    let mut a = real_signal::<R>(1000, 21, 0.01);
    let mut b = real_signal::<R>(1000, 22, 0.03);
    for (i, (x, y)) in [(-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (0.5, -1.0)]
        .into_iter()
        .enumerate()
    {
        a[i] = R::from_f64_round(x);
        b[i] = R::from_f64_round(y);
    }
    let mut sum = vec![R::default(); a.len()];
    let mut prod = vec![R::default(); a.len()];
    Sdrk.add_real(&a, &b, &mut sum).unwrap();
    Sdrk.mul_real(&a, &b, &mut prod).unwrap();
    for plan in std::iter::once(None).chain(plans(41).into_iter().map(Some)) {
        let mut add = AddRealBlock::<R, _>::new(rate(48_000), &mut VecStorage).unwrap();
        let mut mul = MulRealBlock::<R, _>::new(rate(48_000), &mut VecStorage).unwrap();
        assert_eq!(bytes(&run_two(&mut add, &a, &b, plan)), bytes(&sum));
        assert_eq!(bytes(&run_two(&mut mul, &a, &b, plan)), bytes(&prod));
    }
    assert_lockstep(
        &mut AddRealBlock::<R, _>::new(rate(1), &mut VecStorage).unwrap(),
        &a,
        &b,
    );
    assert_lockstep(
        &mut MulRealBlock::<R, _>::new(rate(1), &mut VecStorage).unwrap(),
        &a,
        &b,
    );
}

#[test]
fn add_and_mul_equal_the_kernels_and_keep_lockstep() {
    complex_math_equals_kernels::<C32>();
    complex_math_equals_kernels::<CQ15>();
    real_math_equals_kernels::<f32>();
    real_math_equals_kernels::<Q15>();
}

#[test]
fn the_q15_blocks_saturate_and_follow_the_rounding_rule() {
    let mut mul = MulRealBlock::<Q15, _>::new(rate(1), &mut VecStorage).unwrap();
    let mut out = [Q15(0); 2];
    mul.process(
        (&[Q15(-32768), Q15(16384)], &[Q15(-32768), Q15(16384)]),
        &mut out,
    )
    .unwrap();
    assert_eq!(out, [Q15(32767), Q15(8192)]);
    let mut add = AddRealBlock::<Q15, _>::new(rate(1), &mut VecStorage).unwrap();
    add.process(
        (&[Q15(-32768), Q15(30000)], &[Q15(-1), Q15(30000)]),
        &mut out,
    )
    .unwrap();
    assert_eq!(out, [Q15(-32768), Q15(32767)]);
}

#[test]
fn a_short_output_limits_the_pairs_taken() {
    let mut add = AddBlock::<C32, _>::new(rate(1), &mut VecStorage).unwrap();
    let a = [C32::new(1.0, 1.0); 5];
    let mut out = [C32::default(); 3];
    let p = add.process((&a, &a), &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (3, 3));
    let p = add.process((&[], &[]), &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
}

// ---------------------------------------------------------------------------------------------
// make_complex, to_complex, real_part, imag_part

fn split_round_trip<T: BlockSample>(re: &[T::Real], im: &[T::Real])
where
    Sdrk: Interleave<T>,
{
    let r = rate(48_000);
    for plan in std::iter::once(None).chain(plans(51).into_iter().map(Some)) {
        let mut make = MakeComplexBlock::<T, _>::new(r, &mut VecStorage).unwrap();
        let c: Vec<T> = run_two(&mut make, re, im, plan);
        let mut want = vec![T::default(); re.len()];
        Sdrk.interleave(re, Some(im), &mut want).unwrap();
        assert_eq!(bytes(&c), bytes(&want));
        let mut rp = RealPartBlock::<T, _>::new(r, &mut VecStorage).unwrap();
        let mut ip = ImagPartBlock::<T, _>::new(r, &mut VecStorage).unwrap();
        assert_eq!(bytes(&run_block(&mut rp, &c, plan)), bytes(re));
        assert_eq!(bytes(&run_block(&mut ip, &c, plan)), bytes(im));
        // A real signal made complex keeps its real part and has a zero imaginary part.
        let mut to = ToComplexBlock::<T, _>::new(r, &mut VecStorage).unwrap();
        let z: Vec<T> = run_block(&mut to, re, plan);
        assert_eq!(bytes(&run_block(&mut rp, &z, plan)), bytes(re));
        let zero = vec![<T::Real as Default>::default(); re.len()];
        assert_eq!(bytes(&run_block(&mut ip, &z, plan)), bytes(&zero));
    }
}

#[test]
fn make_complex_and_the_parts_return_the_inputs_bit_for_bit() {
    let re = real_signal::<f32>(700, 61, 0.02);
    let mut im = real_signal::<f32>(700, 62, 0.05);
    // NaN with a payload, infinities and negative zero survive unchanged.
    im[0] = f32::from_bits(0x7fc0_1234);
    im[1] = f32::INFINITY;
    im[2] = -0.0;
    im[3] = f32::NEG_INFINITY;
    split_round_trip::<C32>(&re, &im);
    let re = real_signal::<Q15>(700, 63, 0.02);
    let mut im = real_signal::<Q15>(700, 64, 0.05);
    im[0] = Q15(-32768);
    im[1] = Q15(32767);
    split_round_trip::<CQ15>(&re, &im);
}

#[test]
fn make_complex_rejects_unequal_inputs() {
    let a = vec![0.5f32; 8];
    let mut b = MakeComplexBlock::<C32, _>::new(rate(1), &mut VecStorage).unwrap();
    assert_lockstep(&mut b, &a, &a);
}

// ---------------------------------------------------------------------------------------------
// stereo_matrix

fn matrix_equals_kernel<T: BlockSample>()
where
    Sdrk: StereoMatrix<T>,
{
    let mut input = signal::<T>(900, 71, 0.02);
    input[0] = corner(1.0, 1.0);
    input[1] = corner(-1.0, -1.0);
    let mut want = vec![T::default(); input.len()];
    Sdrk.stereo_matrix(&input, &mut want).unwrap();
    for plan in std::iter::once(None).chain(plans(72).into_iter().map(Some)) {
        let mut b = StereoMatrixBlock::<T, _>::new(rate(48_000), &mut VecStorage).unwrap();
        assert_eq!(bytes(&run_block(&mut b, &input, plan)), bytes(&want));
    }
}

#[test]
fn stereo_matrix_equals_the_kernel() {
    matrix_equals_kernel::<C32>();
    matrix_equals_kernel::<CQ15>();
}

// ---------------------------------------------------------------------------------------------
// nco

fn nco_cfg(frequency_hz: f64) -> NcoConfig {
    NcoConfig {
        frequency_hz,
        ..NcoConfig::default()
    }
}

fn nco_equals_kernel<T: BlockSample>()
where
    Sdrk: Oscillate<T>,
{
    let fs = rate(192_000);
    let clock = signal::<T>(3000, 81, 0.01);
    let mut nco = Nco::new(19_000.0, fs).unwrap();
    nco.set_phase(0x1234_5678);
    let mut want = vec![T::default(); clock.len()];
    Sdrk.oscillate(&mut nco, 3, &mut want).unwrap();
    for plan in std::iter::once(None).chain(plans(82).into_iter().map(Some)) {
        let cfg = NcoConfig {
            frequency_hz: 19_000.0,
            harmonic: 3,
            phase_turns: f64::from(0x1234_5678u32) / 4_294_967_296.0,
        };
        let mut b = NcoBlock::<T, _>::new(&cfg, fs, &mut VecStorage).unwrap();
        assert_eq!(bytes(&run_block(&mut b, &clock, plan)), bytes(&want));
    }
}

#[test]
fn nco_equals_the_oscillate_kernel() {
    nco_equals_kernel::<C32>();
    nco_equals_kernel::<CQ15>();
}

#[test]
fn a_real_clock_gives_the_same_tone_and_its_values_are_ignored() {
    let fs = rate(48_000);
    let cfg = nco_cfg(3_000.0);
    let clock = signal::<C32>(500, 91, 0.01);
    let want = run_block(
        &mut NcoBlock::<C32, _>::new(&cfg, fs, &mut VecStorage).unwrap(),
        &clock,
        None,
    );
    let mut audio = real_signal::<f32>(500, 92, 0.01);
    audio[0] = f32::NAN;
    audio[1] = f32::INFINITY;
    let got = run_block(
        &mut NcoRealBlock::<C32, _>::new(&cfg, fs, &mut VecStorage).unwrap(),
        &audio,
        None,
    );
    assert_eq!(bytes(&got), bytes(&want));
    let q = run_block(
        &mut NcoRealBlock::<CQ15, _>::new(&cfg, fs, &mut VecStorage).unwrap(),
        &real_signal::<Q15>(500, 93, 0.01),
        None,
    );
    let qwant = run_block(
        &mut NcoBlock::<CQ15, _>::new(&cfg, fs, &mut VecStorage).unwrap(),
        &signal::<CQ15>(500, 94, 0.01),
        None,
    );
    assert_eq!(bytes(&q), bytes(&qwant));
}

#[test]
fn a_frequency_change_keeps_the_phase() {
    let fs = rate(48_000);
    let clock = signal::<C32>(100, 95, 0.01);
    let mut b = NcoBlock::<C32, _>::new(&nco_cfg(1_000.0), fs, &mut VecStorage).unwrap();
    let mut got = run_block(&mut b, &clock, None);
    b.set_param("frequency_hz", ParamValue::F64(-7_500.0))
        .unwrap();
    assert_eq!(b.param("frequency_hz"), Ok(ParamValue::F64(-7_500.0)));
    got.extend(run_block(&mut b, &clock, None));

    let mut nco = Nco::new(1_000.0, fs).unwrap();
    let mut want = vec![C32::default(); 200];
    Sdrk.oscillate(&mut nco, 1, &mut want[..100]).unwrap();
    nco.set_frequency(-7_500.0, fs).unwrap();
    Sdrk.oscillate(&mut nco, 1, &mut want[100..]).unwrap();
    assert_eq!(bytes(&got), bytes(&want));
}

#[test]
fn harmonic_two_is_the_square_of_harmonic_one_on_the_same_clock() {
    let fs = rate(192_000);
    let clock = vec![C32::default(); 100_000];
    let one = nco_cfg(19_000.0);
    let two = NcoConfig { harmonic: 2, ..one };
    let a = run_block(
        &mut NcoBlock::<C32, _>::new(&one, fs, &mut VecStorage).unwrap(),
        &clock,
        None,
    );
    let b = run_block(
        &mut NcoBlock::<C32, _>::new(&two, fs, &mut VecStorage).unwrap(),
        &clock,
        None,
    );
    for (x, y) in a.iter().zip(&b) {
        let sq = Complex64::new(f64::from(x.re), f64::from(x.im)).powi(2);
        assert!((sq.re - f64::from(y.re)).abs() < 1e-6 && (sq.im - f64::from(y.im)).abs() < 1e-6);
    }
}

#[test]
fn the_start_phase_and_reset() {
    let fs = rate(48_000);
    let cfg = NcoConfig {
        frequency_hz: 1_000.0,
        harmonic: 1,
        phase_turns: 0.25,
    };
    let mut b = NcoBlock::<C32, _>::new(&cfg, fs, &mut VecStorage).unwrap();
    let clock = vec![C32::default(); 64];
    let first = run_block(&mut b, &clock, None);
    assert!(first[0].re.abs() < 1e-6 && (first[0].im - 1.0).abs() < 1e-6);
    let second = run_block(&mut b, &clock, None);
    assert_ne!(bytes(&first), bytes(&second));
    b.reset();
    assert_eq!(bytes(&run_block(&mut b, &clock, None)), bytes(&first));
    // One full turn is the same start as none.
    let full = NcoConfig {
        phase_turns: 1.0,
        ..cfg
    };
    let zero = NcoConfig {
        phase_turns: 0.0,
        ..cfg
    };
    let a = run_block(
        &mut NcoBlock::<C32, _>::new(&full, fs, &mut VecStorage).unwrap(),
        &clock,
        None,
    );
    let z = run_block(
        &mut NcoBlock::<C32, _>::new(&zero, fs, &mut VecStorage).unwrap(),
        &clock,
        None,
    );
    assert_eq!(bytes(&a), bytes(&z));
}

#[test]
fn the_nco_validates_its_parameters() {
    let fs = rate(48_000);
    let new = |cfg: &NcoConfig| NcoBlock::<C32, _>::new(cfg, fs, &mut VecStorage).map(|_| ());
    assert!(new(&nco_cfg(23_999.0)).is_ok());
    assert!(new(&nco_cfg(-23_999.0)).is_ok());
    for f in [24_000.0, -24_000.0, 1e9, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(new(&nco_cfg(f)), Err(Error::OutOfRange { .. })),
            "{f}"
        );
    }
    for h in [0, 9] {
        let cfg = NcoConfig {
            harmonic: h,
            ..nco_cfg(1.0)
        };
        assert!(matches!(
            new(&cfg),
            Err(Error::OutOfRange {
                name: "harmonic",
                ..
            })
        ));
    }
    for t in [-0.1, 1.1, f64::NAN] {
        let cfg = NcoConfig {
            phase_turns: t,
            ..nco_cfg(1.0)
        };
        assert!(matches!(
            new(&cfg),
            Err(Error::OutOfRange {
                name: "phase_turns",
                ..
            })
        ));
    }
    let mut b = NcoBlock::<C32, _>::new(&nco_cfg(1_000.0), fs, &mut VecStorage).unwrap();
    assert!(matches!(
        b.set_param("frequency_hz", ParamValue::F64(24_000.0)),
        Err(Error::OutOfRange { .. })
    ));
    assert_eq!(b.param("frequency_hz"), Ok(ParamValue::F64(1_000.0)));
    assert_eq!(
        b.set_param("harmonic", ParamValue::I64(2)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(b.param("harmonic"), Ok(ParamValue::I64(1)));
    assert_eq!(b.min_input(), 1);
    assert_eq!(b.timing().ratio, sdrk::Ratio::ONE);
}

// ---------------------------------------------------------------------------------------------
// stereo_pll

const FS: u64 = 480_000;

fn mpx_signal<T: ComplexSample>(n: usize, pilot: f64) -> Vec<T> {
    let tones = stereo_mpx_tones((1.0, 1000.0), (1.0, 3000.0), pilot);
    let mut buf = vec![Complex64::new(0.0, 0.0); n];
    fm_multitone_baseband(&mut buf, FS as f64, 0.0, 75_000.0, &tones, 0).unwrap();
    buf.iter().map(|c| T::from_c64_round(*c * 0.9)).collect()
}

/// The multiplex signal and the pilot band-pass output exactly as `fm_stereo_demod` makes them.
fn pll_inputs<T>(cfg: &FmStereoDemodConfig, input: &[T]) -> (Vec<T::Real>, Vec<T::Real>)
where
    T: BlockSample + BiquadElem,
    T::Real: BiquadElem,
    Sdrk: QuadDemod<T> + BiquadReal<T::Real>,
{
    let fs = FS as f64;
    let mut quad = QuadDemodState::<T>::new();
    let mut mpx = vec![T::Real::default(); input.len()];
    Sdrk.quad_demod(
        &mut quad,
        input,
        &mut mpx,
        T::demod_gain(cfg.deviation_hz, fs),
    )
    .unwrap();
    let bp = biquad_coefs(BiquadResponse::Bandpass, 19_000.0, 5.0, fs).unwrap();
    let mut coefs = vec![Default::default(); 1];
    <T::Real as BiquadElem>::convert(&[bp], &mut coefs).unwrap();
    let mut st = BiquadRealState::<T::Real, _, _>::new(coefs, vec![T::Real::default(); 4]).unwrap();
    let mut pilot = vec![T::Real::default(); input.len()];
    Sdrk.biquad_real(&mut st, &mpx, &mut pilot).unwrap();
    (mpx, pilot)
}

fn pll_level<T: BiquadElem>(cfg: &FmStereoDemodConfig) -> f64 {
    if <T as BiquadElem>::Q15 {
        0.1 * 2.0 * cfg.deviation_hz / FS as f64
    } else {
        0.1
    }
}

fn pll_config<T: BiquadElem>(cfg: &FmStereoDemodConfig) -> StereoPllConfig {
    StereoPllConfig {
        pilot_hz: 19_000.0,
        natural_hz: cfg.pll_natural_hz,
        pilot_level: pll_level::<T>(cfg),
        mono: false,
    }
}

fn stereo_pll_cases<T>()
where
    T: BlockSample + BiquadElem,
    T::Real: BiquadElem,
    Sdrk: QuadDemod<T>
        + BiquadReal<T::Real>
        + StereoPll<T::Real>
        + FirDecimate<T>
        + StereoMatrix<T>
        + Biquad<T>,
{
    let fs = rate(FS);
    let fm_cfg = FmStereoDemodConfig {
        deemphasis_us: 0.0,
        audio_decimation: 5,
        ..FmStereoDemodConfig::default()
    };
    let input = mpx_signal::<T>(48_000, 0.1);
    let (mpx, pilot) = pll_inputs::<T>(&fm_cfg, &input);
    let cfg = pll_config::<T>(&fm_cfg);

    // The block equals the kernel, whole and in chunks.
    let mut state =
        StereoPllState::<T::Real>::new(cfg.pilot_hz, cfg.natural_hz, cfg.pilot_level, fs).unwrap();
    let mut want = vec![T::default(); mpx.len()];
    Sdrk.stereo_pll(&mut state, &mpx, &pilot, &mut want)
        .unwrap();
    for plan in std::iter::once(None).chain(plans(101).into_iter().map(Some)) {
        let mut b = StereoPllBlock::<T, _>::new(&cfg, fs, &mut VecStorage).unwrap();
        let got = run_two(&mut b, &mpx, &pilot, plan);
        assert_eq!(bytes(&got), bytes(&want));
        assert_eq!(b.pilot_locked(), state.is_stereo());
    }
    assert!(state.is_stereo(), "the loop did not lock");

    // The rest of the stereo decoder on the output of the block gives the output of
    // `fm_stereo_demod` bit for bit: the block is the loop that block uses internally.
    let spec = DesignSpec {
        cutoff_hz: fm_cfg.audio_cutoff_hz,
        transition_hz: fm_cfg.audio_cutoff_hz / 4.0,
        attenuation_db: 60.0,
    };
    let n = spec.num_taps(FS as f64, 1023).unwrap();
    let mut design = vec![0.0f64; n];
    spec.design(FS as f64, 1023, &mut design).unwrap();
    let taps: Vec<T::Real> = design
        .iter()
        .map(|v| <T as TunedElem>::tap(*v).unwrap())
        .collect();
    let hist = vec![T::default(); taps.len() - 1];
    let mut fir = FirDecimState::new(taps, hist, fm_cfg.audio_decimation).unwrap();
    let mut dec = vec![T::default(); want.len() / fm_cfg.audio_decimation + 2];
    let p = Sdrk.fir_decimate(&mut fir, &want, &mut dec).unwrap();
    dec.truncate(p.produced);
    let mut lr = vec![T::default(); dec.len()];
    Sdrk.stereo_matrix(&dec, &mut lr).unwrap();
    let mut demod = FmStereoDemodBlock::<T, _>::new(&fm_cfg, fs, &mut VecStorage).unwrap();
    let audio = run_block(&mut demod, &input, None);
    assert_eq!(bytes(&audio), bytes(&lr));
    assert_eq!(demod.pilot_locked(), state.is_stereo());
}

#[test]
fn stereo_pll_equals_the_kernel_and_the_loop_of_fm_stereo_demod() {
    stereo_pll_cases::<C32>();
    stereo_pll_cases::<CQ15>();
}

#[test]
fn stereo_pll_rejects_unequal_inputs_and_keeps_its_state() {
    let fs = rate(FS);
    let cfg = StereoPllConfig::default();
    let mpx = real_signal::<f32>(64, 111, 0.03);
    let pilot = real_signal::<f32>(64, 112, 0.04);
    let mut b = StereoPllBlock::<C32, _>::new(&cfg, fs, &mut VecStorage).unwrap();
    assert_lockstep(&mut b, &mpx, &pilot);
    let mut fresh = StereoPllBlock::<C32, _>::new(&cfg, fs, &mut VecStorage).unwrap();
    assert_eq!(
        bytes(&run_two(&mut b, &mpx, &pilot, None)),
        bytes(&run_two(&mut fresh, &mpx, &pilot, None))
    );
}

#[test]
fn the_stereo_parameter_silences_the_difference_without_unlocking() {
    let fs = rate(FS);
    let fm_cfg = FmStereoDemodConfig::default();
    let input = mpx_signal::<C32>(48_000, 0.1);
    let (mpx, pilot) = pll_inputs::<C32>(&fm_cfg, &input);
    let mut b =
        StereoPllBlock::<C32, _>::new(&StereoPllConfig::default(), fs, &mut VecStorage).unwrap();
    assert_eq!(b.param("stereo"), Ok(ParamValue::Choice(0)));
    let auto = run_two(&mut b, &mpx[..24_000], &pilot[..24_000], None);
    assert!(b.pilot_locked());
    assert!(auto[12_000..].iter().any(|v| v.im != 0.0));
    b.set_param("stereo", ParamValue::Choice(1)).unwrap();
    assert_eq!(b.param("stereo"), Ok(ParamValue::Choice(1)));
    let mono = run_two(&mut b, &mpx[24_000..], &pilot[24_000..], None);
    assert!(b.pilot_locked());
    assert!(mono.iter().all(|v| v.im == 0.0));
    assert!(mono.iter().zip(&mpx[24_000..]).all(|(v, m)| v.re == *m));
    // Reset clears the lock.
    b.reset();
    assert!(!b.pilot_locked());
    assert_eq!(b.param("stereo"), Ok(ParamValue::Choice(1)));
}

#[test]
fn stereo_pll_validates_its_parameters() {
    let fs = rate(FS);
    let new =
        |cfg: &StereoPllConfig| StereoPllBlock::<C32, _>::new(cfg, fs, &mut VecStorage).map(|_| ());
    assert!(new(&StereoPllConfig::default()).is_ok());
    let base = StereoPllConfig::default();
    for cfg in [
        StereoPllConfig {
            natural_hz: 191.0,
            ..base
        },
        StereoPllConfig {
            natural_hz: 0.5,
            ..base
        },
        StereoPllConfig {
            pilot_level: 0.0,
            ..base
        },
        StereoPllConfig {
            pilot_level: 1.5,
            ..base
        },
        StereoPllConfig {
            pilot_hz: f64::NAN,
            ..base
        },
    ] {
        assert!(
            matches!(new(&cfg), Err(Error::OutOfRange { .. })),
            "{cfg:?}"
        );
    }
    // The pilot must be below a quarter of the sample rate; the loop must be slow enough.
    assert!(StereoPllBlock::<C32, _>::new(&base, rate(60_000), &mut VecStorage).is_err());
    let slow_pilot = StereoPllConfig {
        pilot_hz: 1_000.0,
        natural_hz: 20.0,
        ..base
    };
    assert!(new(&slow_pilot).is_err());
    assert!(StereoPllBlock::<C32, VecStorage>::storage_bytes(&slow_pilot, fs).is_err());
    assert!(StereoPllBlock::<C32, VecStorage>::rate_out(&base, rate(60_000)).is_err());
    // A level the Q15 loop cannot represent is an error at construction.
    let tiny = StereoPllConfig {
        pilot_level: 1e-6,
        ..base
    };
    assert!(StereoPllBlock::<CQ15, _>::new(&tiny, fs, &mut VecStorage).is_err());
    assert_eq!(
        StereoPllBlock::<C32, VecStorage>::rate_out(&base, fs),
        Ok(fs)
    );
}

// ---------------------------------------------------------------------------------------------
// storage, descriptions, Send

#[test]
fn the_blocks_need_no_storage() {
    let r = rate(FS);
    let nco = nco_cfg(1_000.0);
    let pll = StereoPllConfig::default();
    assert_eq!(AddBlock::<C32, VecStorage>::storage_bytes(r), Ok(0));
    assert_eq!(MulRealBlock::<Q15, VecStorage>::storage_bytes(r), Ok(0));
    assert_eq!(MakeComplexBlock::<C32, VecStorage>::storage_bytes(r), Ok(0));
    assert_eq!(
        StereoMatrixBlock::<CQ15, VecStorage>::storage_bytes(r),
        Ok(0)
    );
    assert_eq!(NcoBlock::<C32, VecStorage>::storage_bytes(&nco, r), Ok(0));
    assert_eq!(
        StereoPllBlock::<C32, VecStorage>::storage_bytes(&pll, r),
        Ok(0)
    );
    check_exact_storage(0, |a| AddBlock::<C32, _>::new(r, a).map(|_| ()));
    check_exact_storage(0, |a| AddRealBlock::<f32, _>::new(r, a).map(|_| ()));
    check_exact_storage(0, |a| MulBlock::<CQ15, _>::new(r, a).map(|_| ()));
    check_exact_storage(0, |a| MulRealBlock::<Q15, _>::new(r, a).map(|_| ()));
    check_exact_storage(0, |a| MakeComplexBlock::<CQ15, _>::new(r, a).map(|_| ()));
    check_exact_storage(0, |a| ToComplexBlock::<C32, _>::new(r, a).map(|_| ()));
    check_exact_storage(0, |a| RealPartBlock::<C32, _>::new(r, a).map(|_| ()));
    check_exact_storage(0, |a| ImagPartBlock::<CQ15, _>::new(r, a).map(|_| ()));
    check_exact_storage(0, |a| StereoMatrixBlock::<C32, _>::new(r, a).map(|_| ()));
    check_exact_storage(0, |a| NcoBlock::<C32, _>::new(&nco, r, a).map(|_| ()));
    check_exact_storage(0, |a| NcoRealBlock::<CQ15, _>::new(&nco, r, a).map(|_| ()));
    check_exact_storage(0, |a| {
        StereoPllBlock::<CQ15, _>::new(&pll, r, a).map(|_| ())
    });
}

#[test]
fn the_blocks_have_ratio_one_and_no_delay_and_are_send_and_sync() {
    fn assert_send<T: Send + Sync>() {}
    assert_send::<AddBlock<C32, VecStorage>>();
    assert_send::<AddRealBlock<Q15, VecStorage>>();
    assert_send::<MulBlock<CQ15, VecStorage>>();
    assert_send::<MulRealBlock<f32, VecStorage>>();
    assert_send::<MakeComplexBlock<C32, VecStorage>>();
    assert_send::<ToComplexBlock<C32, VecStorage>>();
    assert_send::<RealPartBlock<C32, VecStorage>>();
    assert_send::<ImagPartBlock<C32, VecStorage>>();
    assert_send::<StereoMatrixBlock<C32, VecStorage>>();
    assert_send::<NcoBlock<C32, VecStorage>>();
    assert_send::<NcoRealBlock<C32, VecStorage>>();
    assert_send::<StereoPllBlock<C32, VecStorage>>();
    assert_send::<NcoConfig>();
    assert_send::<StereoPllConfig>();
    let b = AddBlock::<C32, _>::new(rate(1), &mut VecStorage).unwrap();
    let t = b.timing();
    assert_eq!(t.ratio, sdrk::Ratio::ONE);
    assert_eq!(t.delay, sdrk::Delay::ZERO);
    assert_eq!(b.min_input(), 1);
    assert_eq!(b.description().kind, "add");
    assert_eq!(b.param("x"), Err(Error::UnknownParameter));
}
