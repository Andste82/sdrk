//! The blocks `halfband` and `real_to_complex` and the real variant of `convert`: design,
//! timing, alias rejection, block-size invariance, storage, parameters and validation.
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
    clippy::similar_names,
    clippy::needless_pass_by_value,
    clippy::type_complexity,
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

mod common;

use common::{check_exact_storage, rate, real_signal, run_block, signal};
use num_complex::Complex64;
use sdrk::{C32, CQ15, ComplexSample, Delay, Q15, Sdrk};
use sdrk_blocks::design::{halfband_len, halfband_taps, kaiser_beta};
use sdrk_blocks::{
    ArenaStorage, Block, BlockReal, BlockSample, ConvertBlock, ConvertConfig, ConvertRealBlock,
    Error, HalfbandBlock, HalfbandConfig, HalfbandRealBlock, ParamValue, RealToComplexBlock,
    all_descriptions,
};
use sdrk_core::kernels::convert::ConvertReal;
use sdrk_core::kernels::halfband::{HalfbandDecimate, HalfbandDecimateReal, RealToComplex};
use sdrk_device::formats;
use sdrk_testkit::{ChunkPlan, Pcg32, VecStorage};

type A<'a> = ArenaStorage<'a>;

const FS: u64 = 48_000;

fn cfg() -> HalfbandConfig {
    HalfbandConfig::default()
}

/// Identical bits (R6.2: the same backend gives the same bits for any split of the input).
fn same_bits<T: bytemuck::Pod>(a: &[T], b: &[T]) -> bool {
    bytemuck::cast_slice::<T, u8>(a) == bytemuck::cast_slice::<T, u8>(b)
}

fn plans20() -> Vec<ChunkPlan> {
    (0..20)
        .map(|i| ChunkPlan::new(0x4d23 + i, [1, 2, 3, 5, 7, 64, 700, 4097][i as usize % 8]))
        .collect()
}

// ------------------------------------------------------------------ design and taps

/// The response of the block to an impulse at input index 1 is `h[2 j - 1]`; at index 0 it is
/// `h[2 j]`. Together they read out all the taps.
fn impulse_taps<B: Block>(block: &mut B, one: B::In, zero: B::In, n: usize) -> Vec<B::Out>
where
    B::Out: Default,
{
    let mut x = vec![zero; n + 1];
    x[0] = one;
    run_block(block, &x, None)
}

#[test]
fn the_f32_taps_are_the_designed_halfband_taps() {
    let r = rate(FS);
    let c = HalfbandConfig {
        transition_hz: Some(4800.0),
        attenuation_db: 70.0,
        max_taps: 1023,
    };
    let n = halfband_len(4800.0, 70.0, FS as f64).unwrap();
    assert_eq!(c.num_taps(r).unwrap(), n);
    let mut h = vec![0.0f64; n];
    halfband_taps(kaiser_beta(70.0), &mut h).unwrap();

    let mut real = HalfbandRealBlock::<f32, _>::new(&c, r, &mut VecStorage).unwrap();
    assert_eq!(real.num_taps(), n);
    // Even taps: y[j] = h[2 j] for an impulse at index 0, the odd ones for an impulse at 1.
    let even = impulse_taps(&mut real, 1.0f32, 0.0, n);
    real.reset();
    let mut x = vec![0.0f32; n + 2];
    x[1] = 1.0;
    let odd = run_block(&mut real, &x, None);
    for (j, v) in even.iter().enumerate() {
        if 2 * j < n {
            assert_eq!(*v, h[2 * j] as f32, "h[{}]", 2 * j);
        }
    }
    for (j, v) in odd.iter().enumerate().skip(1) {
        if 2 * j - 1 < n {
            assert_eq!(*v, h[2 * j - 1] as f32, "h[{}]", 2 * j - 1);
        }
    }
    // Half the taps are exactly zero.
    let zeros = h.iter().filter(|v| **v == 0.0).count();
    assert_eq!(zeros, (n - 3) / 2);
}

#[test]
fn the_q15_blocks_keep_the_zero_taps_exactly_zero() {
    let r = rate(FS);
    let n = cfg().num_taps(r).unwrap();
    let mut block = HalfbandRealBlock::<Q15, _>::new(&cfg(), r, &mut VecStorage).unwrap();
    let mut x = vec![Q15(0); n + 2];
    x[1] = Q15(32767);
    let odd = run_block(&mut block, &x, None);
    // Odd taps at offsets of the centre by an even number are zero: with m odd, those are the
    // odd indices k with |k - m| even and k != m, so the output j = (k + 1) / 2 is zero there.
    let m = (n - 1) / 2;
    for k in (1..n).step_by(2) {
        if k != m && k.abs_diff(m) % 2 == 0 {
            assert_eq!(odd[k.div_ceil(2)], Q15(0), "tap {k}");
        }
    }
    assert!(
        odd[m.div_ceil(2)].0 > 15000,
        "the centre tap is about one half"
    );
}

fn timing_checks<B: Block>(block: &B, taps: usize) {
    let t = block.timing();
    assert_eq!(t.ratio, sdrk::Ratio::new(1, 2).unwrap());
    assert_eq!(
        t.delay.reduced(),
        Delay::new((taps - 1) as u64, 2).unwrap().reduced()
    );
}

#[test]
fn the_group_delay_matches_an_impulse_within_half_a_sample() {
    let r = rate(FS);
    // The impulse at input index 1 lands on the centre tap h[m] at output j = (m + 1) / 2, which
    // is input time 2 j = m + 1: a delay of m = (N - 1) / 2 input samples after the impulse.
    for taps_cfg in [
        cfg(),
        HalfbandConfig {
            attenuation_db: 90.0,
            ..cfg()
        },
    ] {
        let block = HalfbandRealBlock::<f32, _>::new(&taps_cfg, r, &mut VecStorage).unwrap();
        let n = block.num_taps();
        timing_checks(&block, n);
        let mut block = block;
        let mut x = vec![0.0f32; n + 4];
        x[1] = 1.0;
        let y = run_block(&mut block, &x, None);
        let peak = y
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        let measured = (2 * peak) as f64 - 1.0;
        let promised = block.timing().delay.reduced();
        let promised = promised.num() as f64 / promised.den() as f64;
        assert!(
            (measured - promised).abs() <= 0.5,
            "{measured} vs {promised}"
        );
    }
    let c = HalfbandBlock::<C32, _>::new(&cfg(), r, &mut VecStorage).unwrap();
    timing_checks(&c, c.num_taps());
    let q = RealToComplexBlock::<CQ15, _>::new(&cfg(), r, &mut VecStorage).unwrap();
    timing_checks(&q, q.num_taps());
}

// ---------------------------------------------------------------------- alias rejection

/// RMS of a complex tone `a e^{j 2 pi f n / fs}` after the block, relative to the input, with
/// the transient (two filter lengths) skipped.
fn complex_gain<T: BlockSample>(c: &HalfbandConfig, f_hz: f64) -> f64
where
    Sdrk: HalfbandDecimate<T>,
{
    let r = rate(FS);
    let mut block = HalfbandBlock::<T, _>::new(c, r, &mut VecStorage).unwrap();
    let n = 8192;
    let x: Vec<T> = (0..n)
        .map(|i| {
            let p = std::f64::consts::TAU * f_hz / FS as f64 * i as f64;
            T::from_c64_round(Complex64::new(0.5 * p.cos(), 0.5 * p.sin()))
        })
        .collect();
    let y = run_block(&mut block, &x, None);
    let skip = block.num_taps();
    let e: f64 =
        y[skip..].iter().map(|s| s.to_c64().norm_sqr()).sum::<f64>() / (y.len() - skip) as f64;
    e.sqrt() / 0.5
}

#[test]
fn a_tone_in_the_stop_band_is_attenuated_by_the_design_value_minus_3_db() {
    let fs = FS as f64;
    for (att, tr) in [(60.0, 0.1), (60.0, 0.2), (80.0, 0.1)] {
        let c = HalfbandConfig {
            transition_hz: Some(tr * fs),
            attenuation_db: att,
            max_taps: 1023,
        };
        let limit = 10f64.powf(-(att - 3.0) / 20.0);
        // From the stop-band edge to just below half the input rate.
        let lo = fs / 4.0 + tr * fs / 2.0;
        for k in 0..=12 {
            let f = lo + (fs / 2.0 - 0.01 * fs - lo) * f64::from(k) / 12.0;
            let g = complex_gain::<C32>(&c, f);
            assert!(g <= limit, "{att} dB, tr {tr}: {f} Hz gain {g} > {limit}");
        }
        // The pass band is flat.
        let g = complex_gain::<C32>(&c, fs / 4.0 - tr * fs / 2.0 - 0.01 * fs);
        assert!((g - 1.0).abs() < 0.01, "pass band gain {g}");
        // Q15 meets the same bound up to its own noise floor (-90 dB).
        let gq = complex_gain::<CQ15>(&c, lo + 0.05 * fs);
        assert!(gq <= limit.max(10f64.powf(-80.0 / 20.0)), "CQ15 gain {gq}");
    }
}

fn real_gain<R: BlockReal>(c: &HalfbandConfig, f_hz: f64) -> f64
where
    Sdrk: HalfbandDecimateReal<R>,
{
    let mut block = HalfbandRealBlock::<R, _>::new(c, rate(FS), &mut VecStorage).unwrap();
    let n = 8192;
    let x: Vec<R> = (0..n)
        .map(|i| {
            R::from_f64_round(0.5 * (std::f64::consts::TAU * f_hz / FS as f64 * i as f64).sin())
        })
        .collect();
    let y = run_block(&mut block, &x, None);
    let skip = block.num_taps();
    let e: f64 =
        y[skip..].iter().map(|s| s.to_f64().powi(2)).sum::<f64>() / (y.len() - skip) as f64;
    // A real tone of amplitude 0.5 has rms 0.5 / sqrt 2.
    e.sqrt() / (0.5 / std::f64::consts::SQRT_2)
}

#[test]
fn a_real_tone_in_the_stop_band_is_attenuated_too() {
    let fs = FS as f64;
    let c = cfg();
    let limit = 10f64.powf(-57.0 / 20.0);
    // A real tone at f in (fs/4 + t/2, fs/2) aliases to fs/2 - f; the block removes it.
    for f in [0.31 * fs, 0.38 * fs, 0.45 * fs] {
        let g = real_gain::<f32>(&c, f);
        assert!(g <= limit, "{f} Hz gain {g}");
    }
    let g = real_gain::<f32>(&c, 0.1 * fs);
    assert!((g - 1.0).abs() < 0.01, "{g}");
    let gq = real_gain::<Q15>(&c, 0.38 * fs);
    assert!(gq <= 10f64.powf(-80.0 / 20.0).max(limit), "{gq}");
}

#[test]
fn real_to_complex_maps_f_to_f_minus_a_quarter_and_rejects_the_mirror() {
    let fs = FS as f64;
    let c = cfg();
    let limit = 10f64.powf(-57.0 / 20.0);
    let mut block = RealToComplexBlock::<C32, _>::new(&c, rate(FS), &mut VecStorage).unwrap();
    let n = 16384;
    // A real tone at f = 0.33 fs: output frequency f - fs/4 = 0.08 fs (rate fs/2, so 0.16 cycles
    // per output sample), the mirror at -f lands at -0.33 fs - fs/4 and is removed.
    let f = 0.33 * fs;
    let x: Vec<f32> = (0..n)
        .map(|i| (0.5 * (std::f64::consts::TAU * f / fs * i as f64).cos()) as f32)
        .collect();
    let y = run_block(&mut block, &x, None);
    let skip = block.num_taps();
    let y = &y[skip..];
    let at = |g: f64| -> f64 {
        // Single-bin DFT at the normalised frequency g (cycles per output sample).
        let mut acc = Complex64::new(0.0, 0.0);
        for (i, s) in y.iter().enumerate() {
            let p = -std::f64::consts::TAU * g * i as f64;
            acc += s.to_c64() * Complex64::new(p.cos(), p.sin());
        }
        acc.norm_sqr().sqrt() / y.len() as f64
    };
    let wanted = at((f - fs / 4.0) / (fs / 2.0));
    let mirror = at((-f - fs / 4.0) / (fs / 2.0));
    // A real cosine of amplitude 0.5 is 0.25 per sign of frequency.
    assert!((wanted - 0.25).abs() < 0.01, "wanted {wanted}");
    assert!(mirror / wanted <= limit, "mirror {mirror}");
}

// ---------------------------------------------------------------------- invariance

fn invariance_complex<T: BlockSample>()
where
    Sdrk: HalfbandDecimate<T>,
{
    let c = HalfbandConfig {
        transition_hz: Some(0.2 * FS as f64),
        ..cfg()
    };
    let x = signal::<T>(5000, 11, 0.13);
    let whole = run_block(
        &mut HalfbandBlock::<T, _>::new(&c, rate(FS), &mut VecStorage).unwrap(),
        &x,
        None,
    );
    assert_eq!(whole.len(), 2500);
    for plan in plans20() {
        let mut b = HalfbandBlock::<T, _>::new(&c, rate(FS), &mut VecStorage).unwrap();
        let got = run_block(&mut b, &x, Some(plan));
        assert!(same_bits(&whole, &got), "plan {plan:?}");
    }
    // `reset` restarts from the first input sample.
    let mut b = HalfbandBlock::<T, _>::new(&c, rate(FS), &mut VecStorage).unwrap();
    run_block(&mut b, &x[..777], None);
    b.reset();
    assert!(same_bits(&whole, &run_block(&mut b, &x, None)));
}

fn invariance_real<R: BlockReal>()
where
    Sdrk: HalfbandDecimateReal<R>,
{
    let c = HalfbandConfig {
        transition_hz: Some(0.2 * FS as f64),
        ..cfg()
    };
    let x = real_signal::<R>(5001, 12, 0.13);
    let whole = run_block(
        &mut HalfbandRealBlock::<R, _>::new(&c, rate(FS), &mut VecStorage).unwrap(),
        &x,
        None,
    );
    for plan in plans20() {
        let mut b = HalfbandRealBlock::<R, _>::new(&c, rate(FS), &mut VecStorage).unwrap();
        let got = run_block(&mut b, &x, Some(plan));
        assert!(same_bits(&whole, &got), "plan {plan:?}");
    }
    let mut b = HalfbandRealBlock::<R, _>::new(&c, rate(FS), &mut VecStorage).unwrap();
    run_block(&mut b, &x[..501], None);
    b.reset();
    assert!(same_bits(&whole, &run_block(&mut b, &x, None)));
}

fn invariance_r2c<T: BlockSample>()
where
    Sdrk: RealToComplex<T>,
{
    let c = HalfbandConfig {
        transition_hz: Some(0.2 * FS as f64),
        ..cfg()
    };
    let x = real_signal::<T::Real>(5003, 13, 0.31);
    let whole = run_block(
        &mut RealToComplexBlock::<T, _>::new(&c, rate(FS), &mut VecStorage).unwrap(),
        &x,
        None,
    );
    for plan in plans20() {
        let mut b = RealToComplexBlock::<T, _>::new(&c, rate(FS), &mut VecStorage).unwrap();
        let got = run_block(&mut b, &x, Some(plan));
        assert!(same_bits(&whole, &got), "plan {plan:?}");
    }
    // After a reset the counter n restarts at 0, also after an odd number of samples.
    let mut b = RealToComplexBlock::<T, _>::new(&c, rate(FS), &mut VecStorage).unwrap();
    run_block(&mut b, &x[..501], None);
    b.reset();
    assert!(same_bits(&whole, &run_block(&mut b, &x, None)));
}

#[test]
fn the_blocks_are_independent_of_the_chunking() {
    invariance_complex::<C32>();
    invariance_complex::<CQ15>();
    invariance_real::<f32>();
    invariance_real::<Q15>();
    invariance_r2c::<C32>();
    invariance_r2c::<CQ15>();
}

#[test]
fn a_full_output_buffer_holds_the_input_back() {
    let mut b = HalfbandRealBlock::<f32, _>::new(&cfg(), rate(FS), &mut VecStorage).unwrap();
    let x = vec![0.25f32; 100];
    let mut out = [0.0f32; 3];
    let p = b.process(&x, &mut out).unwrap();
    assert_eq!(p.produced, 3);
    assert!(
        p.consumed <= 7 && p.consumed >= 5,
        "consumed {}",
        p.consumed
    );
    let mut none: [f32; 0] = [];
    assert_eq!(b.process(&x, &mut none).unwrap().produced, 0);
}

// ---------------------------------------------------------------------- storage and rules

#[test]
fn the_arena_is_used_exactly() {
    fn exact<T: BlockSample, R: BlockReal>() {
        for (tr, att) in [(None, 60.0), (Some(2000.0), 80.0)] {
            let c = HalfbandConfig {
                transition_hz: tr,
                attenuation_db: att,
                ..cfg()
            };
            let r = rate(FS);
            let bytes = HalfbandBlock::<T, A<'_>>::storage_bytes(&c, r).unwrap();
            assert!(bytes > 0);
            check_exact_storage(bytes, |a| HalfbandBlock::<T, _>::new(&c, r, a).map(|_| ()));
            let bytes = RealToComplexBlock::<T, A<'_>>::storage_bytes(&c, r).unwrap();
            check_exact_storage(bytes, |a| {
                RealToComplexBlock::<T, _>::new(&c, r, a).map(|_| ())
            });
            let bytes = HalfbandRealBlock::<R, A<'_>>::storage_bytes(&c, r).unwrap();
            check_exact_storage(bytes, |a| {
                HalfbandRealBlock::<R, _>::new(&c, r, a).map(|_| ())
            });
        }
    }
    exact::<C32, f32>();
    exact::<CQ15, Q15>();
}

#[test]
fn no_block_is_larger_than_512_bytes_and_all_are_send() {
    fn check<T: Send>(name: &str) {
        let size = std::mem::size_of::<T>();
        assert!(size <= 512, "{name} is {size} bytes");
    }
    check::<HalfbandBlock<C32, A<'static>>>("HalfbandBlock<C32>");
    check::<HalfbandBlock<CQ15, VecStorage>>("HalfbandBlock<CQ15, Vec>");
    check::<HalfbandRealBlock<f32, A<'static>>>("HalfbandRealBlock<f32>");
    check::<HalfbandRealBlock<Q15, VecStorage>>("HalfbandRealBlock<Q15, Vec>");
    check::<RealToComplexBlock<C32, A<'static>>>("RealToComplexBlock<C32>");
    check::<RealToComplexBlock<CQ15, VecStorage>>("RealToComplexBlock<CQ15, Vec>");
    check::<ConvertRealBlock<f32, A<'static>>>("ConvertRealBlock<f32>");
    check::<ConvertRealBlock<Q15, VecStorage>>("ConvertRealBlock<Q15, Vec>");
    fn sync<T: Sync>() {}
    sync::<HalfbandBlock<C32, A<'static>>>();
    sync::<HalfbandRealBlock<Q15, A<'static>>>();
    sync::<RealToComplexBlock<CQ15, A<'static>>>();
    sync::<ConvertRealBlock<f32, A<'static>>>();
}

// ---------------------------------------------------------------- parameters and validation

#[test]
fn the_descriptions_name_the_kinds_and_the_parameters() {
    for kind in ["halfband", "real_to_complex"] {
        let d = all_descriptions().iter().find(|d| d.kind == kind).unwrap();
        let names: Vec<_> = d.params.iter().map(|p| p.name).collect();
        assert_eq!(names, ["transition_hz", "attenuation_db", "max_taps"]);
        assert_eq!(d.required.len(), 0);
        assert!(
            d.params
                .iter()
                .all(|p| p.mutability == sdrk_blocks::Mutability::Build)
        );
    }
}

#[test]
fn parameters_are_read_back_with_the_defaults_resolved() {
    let r = rate(FS);
    let b = HalfbandBlock::<C32, _>::new(&cfg(), r, &mut VecStorage).unwrap();
    assert_eq!(b.param("transition_hz"), Ok(ParamValue::F64(4800.0)));
    assert_eq!(b.param("attenuation_db"), Ok(ParamValue::F64(60.0)));
    assert_eq!(b.param("max_taps"), Ok(ParamValue::I64(1023)));
    assert_eq!(b.param("nope"), Err(Error::UnknownParameter));
    let mut b = b;
    assert_eq!(
        b.set_param("max_taps", ParamValue::I64(10)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(
        b.set_param("nope", ParamValue::I64(10)),
        Err(Error::UnknownParameter)
    );
    let mut c = HalfbandConfig::default();
    c.set("transition_hz", ParamValue::F64(1000.0)).unwrap();
    assert_eq!(c.get("transition_hz"), Ok(ParamValue::F64(1000.0)));
    assert!(matches!(
        c.set("attenuation_db", ParamValue::F64(7.0)),
        Err(Error::OutOfRange { .. })
    ));
    assert_eq!(
        c.set("attenuation_db", ParamValue::I64(7)),
        Err(Error::WrongParamType)
    );
}

#[test]
fn invalid_designs_are_errors() {
    let r = rate(FS);
    // Transition at or above half the input rate, or not positive.
    for t in [24_000.0, 30_000.0, -1.0, 0.0] {
        let c = HalfbandConfig {
            transition_hz: Some(t),
            ..cfg()
        };
        assert!(c.validate(r).is_err(), "{t}");
        assert!(HalfbandBlock::<C32, _>::new(&c, r, &mut VecStorage).is_err());
    }
    // More taps than allowed.
    let c = HalfbandConfig {
        transition_hz: Some(100.0),
        max_taps: 63,
        ..cfg()
    };
    assert!(matches!(
        c.validate(r),
        Err(Error::OutOfRange {
            name: "max_taps",
            ..
        })
    ));
    // The tap count is 4 k + 3 and never above max_taps.
    for att in [20.0, 60.0, 100.0] {
        let c = HalfbandConfig {
            attenuation_db: att,
            ..cfg()
        };
        let n = c.num_taps(r).unwrap();
        assert_eq!(n % 4, 3);
        assert!(n <= 1023);
    }
}

// ----------------------------------------------------------------------- real convert

#[test]
fn convert_with_rs16_equals_convert_real() {
    let mut rng = Pcg32::new(5, 1);
    let bytes: Vec<u8> = (0..2 * 1000).map(|_| (rng.next_u32() >> 8) as u8).collect();
    let r = rate(4_800_000);
    let c = ConvertConfig {
        format: formats::RS16,
    };
    let mut q = ConvertRealBlock::<Q15, _>::new(&c, r, &mut VecStorage).unwrap();
    let mut out = vec![Q15(0); 1000];
    let p = q.process(&bytes, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (2000, 1000));
    let mut want = vec![Q15(0); 1000];
    Sdrk.convert_real(&formats::RS16, &bytes, &mut want)
        .unwrap();
    assert_eq!(out, want);

    let mut f = ConvertRealBlock::<f32, _>::new(&c, r, &mut VecStorage).unwrap();
    let mut outf = vec![0.0f32; 1000];
    f.process(&bytes, &mut outf).unwrap();
    let mut wantf = vec![0.0f32; 1000];
    Sdrk.convert_real(&formats::RS16, &bytes, &mut wantf)
        .unwrap();
    assert_eq!(outf, wantf);
    assert_eq!(outf[0], f32::from(out[0].0) / 32768.0);

    // Chunking: whole samples only, a partial sample stays.
    let mut g = ConvertRealBlock::<Q15, _>::new(&c, r, &mut VecStorage).unwrap();
    let got = run_block(&mut g, &bytes, Some(ChunkPlan::new(3, 9)));
    assert_eq!(got, want);
    assert_eq!(q.timing().ratio, sdrk::Ratio::new(1, 2).unwrap());
    assert_eq!(q.min_input(), 2);
    assert_eq!(ConvertRealBlock::<Q15, A<'_>>::storage_bytes(&c, r), Ok(0));
    assert_eq!(ConvertRealBlock::<Q15, A<'_>>::rate_out(&c, r), Ok(r));
    assert_eq!(q.param("format"), Ok(ParamValue::Format(formats::RS16)));
}

#[test]
fn the_real_and_the_complex_convert_blocks_refuse_each_others_formats() {
    let r = rate(1_000_000);
    let real = ConvertConfig {
        format: formats::RS16,
    };
    assert!(ConvertBlock::<C32, _>::new(&real, r, &mut VecStorage).is_err());
    assert!(
        ConvertRealBlock::<f32, _>::new(&ConvertConfig::default(), r, &mut VecStorage).is_err()
    );
    let planar = ConvertConfig {
        format: formats::CS16_PLANAR,
    };
    assert!(planar.validate(r).is_err());
}
