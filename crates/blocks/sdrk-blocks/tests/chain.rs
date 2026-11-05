//! `Chain`: equivalence with the stages run one after the other, timing, parameters, storage
//! and the port types a chain accepts.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::large_stack_frames
)]

mod common;

use common::{Aligned, rate, signal};
use sdrk::{C32, CQ15, Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_blocks::{
    ArenaStorage, Block, BlockDescription, BlockTiming, Chain, ChainEnd, ConvertBlock,
    ConvertConfig, DdcBlock, DdcConfig, Error, FirBlock, FirConfig, FmDemodBlock, FmDemodConfig,
    Frame, FrameOut, ParamValue, ResampleBlock, ResampleConfig, ScaleBlock, ScaleConfig,
    SpectrumBlock, SpectrumConfig, StreamElem, Two, chain,
};
use sdrk_testkit::{ChunkPlan, Pcg32, VecStorage};

type Vs = VecStorage;

// ---------------------------------------------------------------------------------------
// runners

fn bits<T: StreamElem>(v: &[T]) -> &[u8] {
    bytemuck::cast_slice(v)
}

/// Identical bits; for float outputs also equal up to 120 dB SNR, because the SIMD kernels
/// inside `fm_demod` are tolerance kernels whose bits can depend on how the input is split
/// (maintainer decision of 2026-10-29, see `behaviour.rs`). Q15 outputs must be identical.
fn same_or_close<T: StreamElem>(a: &[T], b: &[T]) -> bool {
    if bits(a) == bits(b) {
        return true;
    }
    let float = core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>()
        || core::any::TypeId::of::<T>() == core::any::TypeId::of::<C32>();
    if !float || a.len() != b.len() {
        return false;
    }
    let (a, b) = (
        bytemuck::cast_slice::<T, f32>(a),
        bytemuck::cast_slice::<T, f32>(b),
    );
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        sig += f64::from(x) * f64::from(x);
        err += (f64::from(x) - f64::from(y)).powi(2);
    }
    10.0 * (sig / err).log10() >= 120.0
}

/// Runs a chain over `input` with input pieces and output sizes from the plans.
fn run_chain<B: Block>(
    chain: &mut B,
    input: &[B::In],
    in_plan: Option<ChunkPlan>,
    out_plan: Option<ChunkPlan>,
) -> Vec<B::Out>
where
    B::In: StreamElem,
    B::Out: StreamElem,
{
    let mut out = Vec::new();
    let mut room = vec![<B::Out as Default>::default(); 4096];
    let mut pending: Vec<B::In> = Vec::new();
    let mut in_sizes = in_plan.map(|p| p.sizes());
    let mut out_sizes = out_plan.map(|p| p.sizes());
    let mut pos = 0;
    while pos < input.len() {
        let want = in_sizes
            .as_mut()
            .map_or(input.len(), |s| s.next().unwrap())
            .min(input.len() - pos);
        pending.extend_from_slice(&input[pos..pos + want]);
        pos += want;
        loop {
            let n = out_sizes
                .as_mut()
                .map_or(room.len(), |s| s.next().unwrap())
                .min(room.len());
            let p = chain
                .process(
                    <B::In as StreamElem>::view(&pending),
                    <B::Out as StreamElem>::view_mut(&mut room[..n]),
                )
                .unwrap();
            out.extend_from_slice(&room[..p.produced]);
            pending.drain(..p.consumed);
            if p.consumed == 0 && p.produced == 0 {
                break;
            }
        }
    }
    out
}

/// Runs a chain that ends in a frame block.
fn frames_chain<B: Block<Out = Frame>>(
    chain: &mut B,
    input: &[B::In],
    plan: Option<ChunkPlan>,
) -> Vec<Vec<u32>>
where
    B::In: StreamElem,
{
    let mut out = Vec::new();
    let mut sizes = plan.map(|p| p.sizes());
    let mut frame = vec![0.0f32; chain.frame_len()];
    let mut pending: Vec<B::In> = Vec::new();
    let mut pos = 0;
    while pos < input.len() {
        let want = sizes
            .as_mut()
            .map_or(input.len(), |s| s.next().unwrap())
            .min(input.len() - pos);
        pending.extend_from_slice(&input[pos..pos + want]);
        pos += want;
        loop {
            let mut meta = None;
            let p = chain
                .process(
                    <B::In as StreamElem>::view(&pending),
                    FrameOut {
                        values: &mut frame,
                        meta: &mut meta,
                    },
                )
                .unwrap();
            if meta.is_some() {
                out.push(frame.iter().map(|v| v.to_bits()).collect());
            }
            pending.drain(..p.consumed);
            if meta.is_none() && p.consumed == 0 {
                break;
            }
        }
    }
    out
}

fn build<A, B>((a, b): (A, B), chunk: usize) -> Chain<A, B, Vs>
where
    A: Block,
    B: Block<In = A::Out>,
    A::In: ChainEnd,
    A::Out: StreamElem,
    B::Out: ChainEnd,
{
    Chain::new(a, b, chunk, &mut VecStorage).unwrap()
}

/// The chain equals the stages one after the other, for five buffer sizes and 20 cut plans.
fn check_equivalence<A, B>(mk: impl Fn() -> (A, B), input: &[A::In], tolerant: bool)
where
    A: Block,
    B: Block<In = A::Out>,
    A::In: StreamElem,
    A::Out: StreamElem,
    B::Out: StreamElem,
{
    let same = |x: &[B::Out], y: &[B::Out]| {
        if tolerant {
            same_or_close(x, y)
        } else {
            bits(x) == bits(y)
        }
    };
    let (mut a, mut b) = mk();
    let min_b = b.min_input();
    let mid = common::run_block(&mut a, input, None);
    let whole = common::run_block(&mut b, &mid, None);
    assert!(!whole.is_empty());
    for chunk in [1, min_b, 7, 256, 4096] {
        let mut c = build(mk(), chunk);
        let got = run_chain(&mut c, input, None, None);
        assert!(same(&got, &whole), "chunk {chunk}");
    }
    for seed in 0..20u64 {
        let mut c = build(mk(), 256);
        let got = run_chain(
            &mut c,
            input,
            Some(ChunkPlan::new(100 + seed, 300)),
            Some(ChunkPlan::new(200 + seed, 50)),
        );
        assert!(same(&got, &whole), "plan {seed}");
    }
}

/// Same for a chain that ends in a frame block.
fn check_frames<A, B>(mk: impl Fn() -> (A, B), input: &[A::In])
where
    A: Block,
    B: Block<In = A::Out, Out = Frame>,
    A::In: StreamElem,
    A::Out: StreamElem,
{
    let (mut a, mut b) = mk();
    let min_b = b.min_input();
    let mid = common::run_block(&mut a, input, None);
    let whole: Vec<Vec<u32>> = common::run_frames(&mut b, &mid, None)
        .into_iter()
        .map(|(v, _)| v.iter().map(|x| x.to_bits()).collect())
        .collect();
    assert!(whole.len() > 2);
    for chunk in [1, min_b, 7, 256, 4096] {
        let mut c = build(mk(), chunk);
        assert_eq!(frames_chain(&mut c, input, None), whole, "chunk {chunk}");
    }
    for seed in 0..20u64 {
        let mut c = build(mk(), 256);
        let got = frames_chain(&mut c, input, Some(ChunkPlan::new(100 + seed, 300)));
        assert_eq!(got.len(), whole.len(), "plan {seed} frames");
        assert!(got == whole, "plan {seed} content");
    }
}

// ---------------------------------------------------------------------------------------
// the stage pairs

macro_rules! pairs {
    ($($t:ident => $m:ident),*) => {$(
        mod $m {
            use super::*;

            pub fn fir_resample() -> (FirBlock<$t, Vs>, ResampleBlock<$t, Vs>) {
                let r = rate(48_000);
                let fir = FirBlock::new(
                    &FirConfig { cutoff_hz: 9_000.0, transition_hz: 3_000.0, ..FirConfig::default() },
                    r,
                    &mut VecStorage,
                )
                .unwrap();
                let res = ResampleBlock::new(
                    &ResampleConfig { rate_out_hz: 32_000.0, ..ResampleConfig::default() },
                    r,
                    &mut VecStorage,
                )
                .unwrap();
                (fir, res)
            }

            pub fn ddc_fm() -> (DdcBlock<$t, Vs>, FmDemodBlock<$t, Vs>) {
                let ddc = DdcBlock::new(
                    &DdcConfig { frequency_hz: 20_000.0, decimation: 4, max_chunk: 128, ..DdcConfig::default() },
                    rate(240_000),
                    &mut VecStorage,
                )
                .unwrap();
                let fm = FmDemodBlock::new(
                    &FmDemodConfig { audio_decimation: 2, audio_cutoff_hz: 10_000.0, ..FmDemodConfig::default() },
                    rate(60_000),
                    &mut VecStorage,
                )
                .unwrap();
                (ddc, fm)
            }

            pub fn convert_spectrum() -> (ConvertBlock<$t, Vs>, SpectrumBlock<$t, Vs>) {
                let conv = ConvertBlock::new(&ConvertConfig::default(), rate(48_000), &mut VecStorage).unwrap();
                let spec = SpectrumBlock::new(
                    &SpectrumConfig { fft_size: 64, averaging: 2, ..SpectrumConfig::default() },
                    rate(48_000),
                    &mut VecStorage,
                )
                .unwrap();
                (conv, spec)
            }

            #[test]
            fn fir_then_resample_equals_the_stages_in_a_row() {
                check_equivalence(fir_resample, &signal::<$t>(6_000, 1, 0.05), false);
            }

            #[test]
            fn ddc_then_fm_demod_equals_the_stages_in_a_row() {
                check_equivalence(ddc_fm, &signal::<$t>(8_000, 2, 0.1), true);
            }

            #[test]
            fn convert_then_spectrum_equals_the_stages_in_a_row() {
                check_frames(convert_spectrum, &cu8(4_000));
            }
        }
    )*};
}

/// Pseudo-random `cu8` bytes (a tone plus noise).
fn cu8(samples: usize) -> Vec<u8> {
    let mut rng = Pcg32::new(5, 1);
    (0..samples * 2)
        .map(|i| {
            let ph = std::f64::consts::TAU * 0.08 * (i / 2) as f64;
            let v = if i % 2 == 0 { ph.cos() } else { ph.sin() };
            (127.5 + 90.0 * v + 4.0 * rng.normal()).clamp(0.0, 255.0) as u8
        })
        .collect()
}

pairs!(C32 => c32, CQ15 => cq15);

// ---------------------------------------------------------------------------------------
// interpolating stages with small output slices

fn resample_fir() -> (ResampleBlock<C32, Vs>, FirBlock<C32, Vs>) {
    let res = ResampleBlock::new(
        &ResampleConfig {
            rate_out_hz: 48_000.0,
            ..ResampleConfig::default()
        },
        rate(8_000),
        &mut VecStorage,
    )
    .unwrap();
    let fir = FirBlock::new(
        &FirConfig {
            cutoff_hz: 9_000.0,
            transition_hz: 3_000.0,
            ..FirConfig::default()
        },
        rate(48_000),
        &mut VecStorage,
    )
    .unwrap();
    (res, fir)
}

fn fir_resample_up() -> (FirBlock<C32, Vs>, ResampleBlock<C32, Vs>) {
    let fir = FirBlock::new(
        &FirConfig {
            cutoff_hz: 3_000.0,
            transition_hz: 1_000.0,
            ..FirConfig::default()
        },
        rate(8_000),
        &mut VecStorage,
    )
    .unwrap();
    let res = ResampleBlock::new(
        &ResampleConfig {
            rate_out_hz: 48_000.0,
            ..ResampleConfig::default()
        },
        rate(8_000),
        &mut VecStorage,
    )
    .unwrap();
    (fir, res)
}

/// Output that a stage still owes when its output slice was full, or when the input is used up,
/// is not lost: the total equals the two stages in a row for every buffer size and small slices.
#[test]
fn interpolating_stages_lose_nothing_with_small_output_slices() {
    let input = signal::<C32>(400, 3, 0.05);
    for plan_max in [5usize, 3] {
        {
            let (mut a, mut b) = resample_fir();
            let mid = common::run_block(&mut a, &input, None);
            let whole = common::run_block(&mut b, &mid, None);
            for chunk in [1, 7, 16, 64] {
                let mut c = build(resample_fir(), chunk);
                let got = run_chain(&mut c, &input, None, Some(ChunkPlan::new(9, plan_max)));
                assert_eq!(bits(&got), bits(&whole), "resample->fir chunk {chunk}");
            }
        }
        {
            let (mut a, mut b) = fir_resample_up();
            let mid = common::run_block(&mut a, &input, None);
            let whole = common::run_block(&mut b, &mid, None);
            for chunk in [1, 7, 16, 64] {
                let mut c = build(fir_resample_up(), chunk);
                let got = run_chain(&mut c, &input, None, Some(ChunkPlan::new(9, plan_max)));
                assert_eq!(bits(&got), bits(&whole), "fir->resample chunk {chunk}");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// timing

/// An exact fraction for the hand computation of the table below.
fn frac_eq(num: u64, den: u64, want_num: u64, want_den: u64) -> bool {
    u128::from(num) * u128::from(want_den) == u128::from(want_num) * u128::from(den)
}

fn assert_timing<A, B>(c: &Chain<A, B, Vs>)
where
    A: Block,
    B: Block<In = A::Out>,
    A::In: ChainEnd,
    A::Out: StreamElem,
    B::Out: ChainEnd,
{
    let (ta, tb) = (c.first().timing(), c.second().timing());
    let t = c.timing();
    // ratio = ratio_a * ratio_b
    assert!(frac_eq(
        t.ratio.num(),
        t.ratio.den(),
        ta.ratio.num() * tb.ratio.num(),
        ta.ratio.den() * tb.ratio.den()
    ));
    // delay = delay_a + delay_b / ratio_a, with delay = n/d: na/da + nb*ra_den/(db*ra_num)
    let (na, da) = (
        u128::from(ta.delay.reduced().num()),
        u128::from(ta.delay.reduced().den()),
    );
    let (nb, db) = (
        u128::from(tb.delay.reduced().num()),
        u128::from(tb.delay.reduced().den()),
    );
    let (rn, rd) = (u128::from(ta.ratio.num()), u128::from(ta.ratio.den()));
    let want_num = na * db * rn + nb * rd * da;
    let want_den = da * db * rn;
    let d = t.delay.reduced();
    assert_eq!(
        u128::from(d.num()) * want_den,
        want_num * u128::from(d.den()),
        "delay"
    );
    // min_input = max(a.min_input, ceil(b.min_input / ratio_a))
    let need = (c.second().min_input() as u128 * rd).div_ceil(rn) as usize;
    assert_eq!(c.min_input(), need.max(c.first().min_input()));
    // map_index composes
    for n in [0u64, 1, 2, 5, 17, 1000, 123_457] {
        assert_eq!(c.map_index(n), c.second().map_index(c.first().map_index(n)));
    }
    assert_eq!(
        t.kind == RatioKind::Variable,
        ta.kind == RatioKind::Variable || tb.kind == RatioKind::Variable
    );
    assert_eq!(c.frame_len(), c.second().frame_len());
}

#[test]
fn timing_min_input_and_map_index_follow_the_stages() {
    let c = build(c32::fir_resample(), 64);
    assert_timing(&c);
    // fir at 48 kHz then 3/2 decimation: ratio 2/3
    assert!(frac_eq(
        c.timing().ratio.num(),
        c.timing().ratio.den(),
        2,
        3
    ));
    assert_eq!(c.timing().kind, RatioKind::Constant);

    let c = build(c32::ddc_fm(), 64);
    assert_timing(&c);
    // ddc by 4, fm_demod by 2: ratio 1/8
    assert!(frac_eq(
        c.timing().ratio.num(),
        c.timing().ratio.den(),
        1,
        8
    ));

    let c = build(cq15::convert_spectrum(), 64);
    assert_timing(&c);
    assert_eq!(c.frame_len(), 64);
    assert_eq!(c.timing().ratio.num(), 1);
}

#[test]
fn the_delay_of_the_second_stage_counts_in_samples_of_the_first() {
    // ddc (decimation 4) then fir: a delay of d samples after the ddc is 4 d samples before.
    let ddc = DdcBlock::<C32, _>::new(
        &DdcConfig {
            decimation: 4,
            max_chunk: 64,
            ..DdcConfig::default()
        },
        rate(240_000),
        &mut VecStorage,
    )
    .unwrap();
    let fir =
        FirBlock::<C32, _>::new(&FirConfig::default(), rate(60_000), &mut VecStorage).unwrap();
    let (da, db) = (ddc.timing().delay, fir.timing().delay);
    let c = build((ddc, fir), 64);
    let want = da
        .add(Delay::new(db.reduced().num() * 4, db.reduced().den()).unwrap())
        .unwrap();
    assert_eq!(c.timing().delay.reduced(), want.reduced());
    assert!(c.timing().delay.as_f64() > da.as_f64());
}

#[test]
fn a_chain_of_three_resamplers_has_the_product_ratio() -> Result<(), Error> {
    let mk = |from: u64, to: f64| {
        ResampleBlock::<C32, _>::new(
            &ResampleConfig {
                rate_out_hz: to,
                ..ResampleConfig::default()
            },
            rate(from),
            &mut VecStorage,
        )
        .unwrap()
    };
    let mut st = VecStorage;
    let c = chain!(
        &mut st, 64;
        mk(48_000, 24_000.0),
        mk(24_000, 16_000.0),
        mk(16_000, 8_000.0)
    );
    let t = c.timing();
    assert!(frac_eq(t.ratio.num(), t.ratio.den(), 1, 6), "{:?}", t.ratio);
    assert_eq!(t.kind, RatioKind::Constant);
    // nested: stage 1 is a chain of the last two
    assert!(frac_eq(
        c.second().timing().ratio.num(),
        c.second().timing().ratio.den(),
        1,
        3
    ));
    assert!(c.map_index(6) >= 1);
    Ok(())
}

// ---------------------------------------------------------------------------------------
// parameters

fn receiver<'a>(
    arena: &mut ArenaStorage<'a>,
) -> Result<impl Block<In = u8, Out = f32> + 'a, Error> {
    let conv = ConvertBlock::<C32, _>::new(&ConvertConfig::default(), rate(240_000), arena)?;
    let ddc = DdcBlock::<C32, _>::new(
        &DdcConfig {
            frequency_hz: 10_000.0,
            decimation: 4,
            max_chunk: 64,
            ..DdcConfig::default()
        },
        rate(240_000),
        arena,
    )?;
    let fm = FmDemodBlock::<C32, _>::new(&FmDemodConfig::default(), rate(60_000), arena)?;
    Ok(chain!(&mut *arena, 64; conv, ddc, fm))
}

#[test]
fn parameters_are_addressed_by_stage_and_nest() {
    let mut mem = Aligned::new(1 << 16);
    let mut arena = ArenaStorage::new(mem.bytes(1 << 16));
    let mut rx = receiver(&mut arena).unwrap();
    // chain(convert, chain(ddc, fm_demod)): the ddc is "1.0", the fm_demod "1.1"
    rx.set_param("1.0.frequency_hz", ParamValue::F64(25_000.0))
        .unwrap();
    assert_eq!(rx.param("1.0.frequency_hz"), Ok(ParamValue::F64(25_000.0)));
    assert_eq!(rx.param("1.1.deviation_hz"), Ok(ParamValue::F64(75_000.0)));
    // no prefix, a wrong stage, a deeper wrong stage
    for name in [
        "frequency_hz",
        "2.frequency_hz",
        "1.2.frequency_hz",
        "0",
        "",
        "1",
        "1.0",
    ] {
        assert_eq!(
            rx.set_param(name, ParamValue::F64(1.0)),
            Err(Error::UnknownParameter),
            "{name:?}"
        );
        assert_eq!(rx.param(name), Err(Error::UnknownParameter), "{name:?}");
    }
    // the stage's own error comes back unchanged
    assert_eq!(
        rx.set_param("0.format", ParamValue::F64(1.0)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(
        rx.set_param("1.0.frequency_hz", ParamValue::I64(1)),
        Err(Error::WrongParamType)
    );
    assert!(matches!(
        rx.set_param("1.0.frequency_hz", ParamValue::F64(1e9)),
        Err(Error::OutOfRange { .. })
    ));
    assert_eq!(rx.param("1.0.frequency_hz"), Ok(ParamValue::F64(25_000.0)));
}

#[test]
fn a_two_stage_chain_reaches_stage_one_as_one_dot() {
    let ddc = DdcBlock::<C32, _>::new(
        &DdcConfig {
            max_chunk: 64,
            ..DdcConfig::default()
        },
        rate(48_000),
        &mut VecStorage,
    )
    .unwrap();
    let mut c = build((scale_first(), ddc), 16);
    c.set_param("1.frequency_hz", ParamValue::F64(1_000.0))
        .unwrap();
    assert_eq!(
        c.second().param("frequency_hz"),
        Ok(ParamValue::F64(1_000.0))
    );
    c.set_param("0.gain_db", ParamValue::F64(-6.0)).unwrap();
    assert_eq!(c.first().param("gain_db"), Ok(ParamValue::F64(-6.0)));
}

fn scale_first() -> ScaleBlock<C32, Vs> {
    ScaleBlock::new(&ScaleConfig::default(), rate(48_000), &mut VecStorage).unwrap()
}

// ---------------------------------------------------------------------------------------
// storage

#[test]
fn storage_bytes_is_exact_on_a_fresh_arena_and_bounded_on_a_shifted_one() {
    for chunk in [1usize, 7, 64, 256] {
        // C32 intermediate samples (8 bytes)
        let mk = || {
            let a = ConvertBlock::<C32, ArenaStorage<'_>>::new(
                &ConvertConfig::default(),
                rate(48_000),
                &mut ArenaStorage::new(&mut []),
            )
            .unwrap();
            let b = ScaleBlock::<C32, ArenaStorage<'_>>::new(
                &ScaleConfig::default(),
                rate(48_000),
                &mut ArenaStorage::new(&mut []),
            )
            .unwrap();
            (a, b)
        };
        let want = Chain::<
            ConvertBlock<C32, ArenaStorage<'_>>,
            ScaleBlock<C32, ArenaStorage<'_>>,
            ArenaStorage<'_>,
        >::storage_bytes(chunk);
        assert_eq!(want, chunk * 8);
        let mut mem = Aligned::new(4096);
        let mut arena = ArenaStorage::new(mem.bytes(4096));
        let (a, b) = mk();
        Chain::new(a, b, chunk, &mut arena).unwrap();
        assert_eq!(arena.used(), want, "fresh arena, chunk {chunk}");

        // one byte in: the address is misaligned for a 4-byte element
        let mut arena = ArenaStorage::new(mem.bytes(4096).split_at_mut(1).1);
        let (a, b) = mk();
        Chain::new(a, b, chunk, &mut arena).unwrap();
        assert_eq!(arena.used(), want + 3, "shifted arena, chunk {chunk}");
    }
}

#[test]
fn a_too_small_arena_is_a_storage_error() {
    let mut mem = Aligned::new(64);
    let mut arena = ArenaStorage::new(mem.bytes(64));
    let a =
        ConvertBlock::<C32, _>::new(&ConvertConfig::default(), rate(48_000), &mut arena).unwrap();
    let b = ScaleBlock::<C32, _>::new(&ScaleConfig::default(), rate(48_000), &mut arena).unwrap();
    // 9 samples of 8 bytes need 72 bytes
    let err = Chain::new(a, b, 9, &mut arena).unwrap_err();
    assert!(
        matches!(
            err,
            Error::Storage {
                needed: 72,
                available: 64
            }
        ),
        "{err:?}"
    );
}

#[test]
fn the_buffer_is_raised_to_what_the_second_stage_needs() {
    let (a, b) = c32::fir_resample();
    let min_b = b.min_input();
    assert!(min_b > 1);
    let mut mem = Aligned::new(1 << 12);
    let mut arena = ArenaStorage::new(mem.bytes(1 << 12));
    let before = arena.used();
    Chain::new(a, b, 1, &mut arena).unwrap();
    assert_eq!(arena.used() - before, min_b * 8);
}

// ---------------------------------------------------------------------------------------
// port types

/// Test-only block: `Two<f32, f32>` (re, im) to `C32`.
struct Join;

impl Block for Join {
    type In = Two<f32, f32>;
    type Out = C32;
    fn description(&self) -> &'static BlockDescription {
        &sdrk_blocks::CHAIN_DESCRIPTION
    }
    fn process(&mut self, input: (&[f32], &[f32]), output: &mut [C32]) -> Result<Progress, Error> {
        if input.0.len() != input.1.len() {
            return Err(Error::InvalidParameter {
                name: "input",
                reason: "length mismatch",
            });
        }
        let n = input.0.len().min(output.len());
        for ((o, &re), &im) in output.iter_mut().zip(input.0).zip(input.1).take(n) {
            *o = C32::new(re, im);
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }
    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: Ratio::new(1, 1).unwrap(),
            kind: RatioKind::Constant,
            delay: Delay::new(0, 1).unwrap(),
        }
    }
    fn reset(&mut self) {}
    fn set_param(&mut self, _: &str, _: ParamValue) -> Result<(), Error> {
        Err(Error::UnknownParameter)
    }
    fn param(&self, _: &str) -> Result<ParamValue, Error> {
        Err(Error::UnknownParameter)
    }
    fn min_input(&self) -> usize {
        1
    }
}

/// Test-only block: `C32` to `Two<f32, f32>`.
struct Split;

impl Block for Split {
    type In = C32;
    type Out = Two<f32, f32>;
    fn description(&self) -> &'static BlockDescription {
        &sdrk_blocks::CHAIN_DESCRIPTION
    }
    fn process(
        &mut self,
        input: &[C32],
        output: (&mut [f32], &mut [f32]),
    ) -> Result<Progress, Error> {
        if output.0.len() != output.1.len() {
            return Err(Error::InvalidParameter {
                name: "output",
                reason: "length mismatch",
            });
        }
        let n = input.len().min(output.0.len());
        for ((x, re), im) in input
            .iter()
            .zip(output.0.iter_mut())
            .zip(output.1.iter_mut())
            .take(n)
        {
            *re = x.re;
            *im = x.im;
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }
    fn timing(&self) -> BlockTiming {
        Join.timing()
    }
    fn reset(&mut self) {}
    fn set_param(&mut self, _: &str, _: ParamValue) -> Result<(), Error> {
        Err(Error::UnknownParameter)
    }
    fn param(&self, _: &str) -> Result<ParamValue, Error> {
        Err(Error::UnknownParameter)
    }
    fn min_input(&self) -> usize {
        1
    }
}

#[test]
fn a_chain_can_start_with_a_pair_of_streams_and_end_with_one() {
    let re: Vec<f32> = (0..500).map(|i| i as f32).collect();
    let im: Vec<f32> = (0..500).map(|i| -(i as f32)).collect();
    let mut c = build((Join, Split), 16);
    let (mut ore, mut oim) = (vec![0.0f32; 500], vec![0.0f32; 500]);
    let (mut done_in, mut done_out) = (0, 0);
    while done_out < 500 {
        let step = (500 - done_out).min(33);
        let p = c
            .process(
                (
                    &re[done_in..(done_in + 40).min(500)],
                    &im[done_in..(done_in + 40).min(500)],
                ),
                (
                    &mut ore[done_out..done_out + step],
                    &mut oim[done_out..done_out + step],
                ),
            )
            .unwrap();
        done_in += p.consumed;
        done_out += p.produced;
        assert!(p.consumed + p.produced > 0);
    }
    assert_eq!(ore, re);
    assert_eq!(oim, im);
}

#[test]
fn unequal_pairs_are_the_stage_error_and_empty_calls_do_nothing() {
    let mut c = build(
        (
            Join,
            ScaleBlock::<C32, Vs>::new(&ScaleConfig::default(), rate(1), &mut VecStorage).unwrap(),
        ),
        8,
    );
    let mut out = [C32::default(); 8];
    let err = c.process((&[1.0, 2.0], &[1.0]), &mut out).unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
    assert_eq!(
        c.process((&[], &[]), &mut out).unwrap(),
        Progress::default()
    );
    assert_eq!(c.process((&[1.0], &[1.0]), &mut []).unwrap().produced, 0);
    // nothing is left half-way: the next call works
    let p = c.process((&[1.0], &[2.0]), &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (1, 1));
    assert_eq!(out[0], C32::new(1.0, 2.0));
}

#[test]
fn reset_empties_the_buffer_and_the_stages() {
    let (a, b) = cq15::ddc_fm();
    let input = signal::<CQ15>(2_000, 9, 0.1);
    let mut c = build((a, b), 64);
    let first = run_chain(&mut c, &input, Some(ChunkPlan::new(1, 50)), None);
    c.reset();
    let second = run_chain(&mut c, &input, Some(ChunkPlan::new(2, 90)), None);
    assert_eq!(bits(&first), bits(&second));
}

#[test]
fn the_calls_per_process_are_bounded() {
    struct Counted<B>(B, usize);
    impl<B: Block<In = C32, Out = C32>> Block for Counted<B> {
        type In = C32;
        type Out = C32;
        fn description(&self) -> &'static BlockDescription {
            self.0.description()
        }
        fn process(&mut self, i: &[C32], o: &mut [C32]) -> Result<Progress, Error> {
            self.1 += 1;
            self.0.process(i, o)
        }
        fn timing(&self) -> BlockTiming {
            self.0.timing()
        }
        fn reset(&mut self) {
            self.0.reset();
        }
        fn set_param(&mut self, n: &str, v: ParamValue) -> Result<(), Error> {
            self.0.set_param(n, v)
        }
        fn param(&self, n: &str) -> Result<ParamValue, Error> {
            self.0.param(n)
        }
        fn min_input(&self) -> usize {
            self.0.min_input()
        }
    }
    let scale = || {
        ScaleBlock::<C32, Vs>::new(
            &ScaleConfig::default(),
            Rate::hz(1).unwrap(),
            &mut VecStorage,
        )
        .unwrap()
    };
    let chunk = 16;
    let mut c = build((Counted(scale(), 0), Counted(scale(), 0)), chunk);
    let input = signal::<C32>(1000, 3, 0.1);
    let mut out = vec![C32::default(); 100];
    let p = c.process(&input, &mut out).unwrap();
    assert_eq!(p.produced, 100);
    let bound = 2 + input.len() / chunk + out.len();
    assert!(
        c.first().1 <= bound && c.second().1 <= bound,
        "{} {} {bound}",
        c.first().1,
        c.second().1
    );
    assert!(c.first().1 < 100, "A is called per buffer, not per sample");
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn chains_are_send_sync_and_describe_themselves() {
    assert_send_sync::<Chain<Join, Split, Vs>>();
    let c = build((Join, Split), 4);
    assert_eq!(c.description().kind, "chain");
    assert!(
        !sdrk_blocks::all_descriptions()
            .iter()
            .any(|d| d.kind == "chain")
    );
}

// ---------------------------------------------------------------------------------------
// a fork as the first stage, a block that reads the pair as the second (M31 test 4)

mod with_fork {
    use super::*;
    use common::{check_exact_storage, real_signal};
    use sdrk_blocks::design::BiquadResponse;
    use sdrk_blocks::{
        BiquadConfig, BiquadRealBlock, DelayConfig, DelayRealBlock, Fork, MakeComplexBlock,
        StereoPllBlock, StereoPllConfig, Storage,
    };

    type As = ArenaStorage<'static>;

    const FS_PLL: u64 = 192_000;

    fn delay_real<S: Storage>(samples: usize, fs: u64, st: &mut S) -> DelayRealBlock<f32, S> {
        DelayRealBlock::new(&DelayConfig { samples }, rate(fs), st).unwrap()
    }

    fn lowpass_cfg() -> BiquadConfig {
        BiquadConfig {
            response: BiquadResponse::Lowpass,
            cutoff_hz: 2_000.0,
            order: 4,
            ..BiquadConfig::default()
        }
    }

    fn pilot_cfg() -> BiquadConfig {
        BiquadConfig {
            response: BiquadResponse::Bandpass,
            cutoff_hz: 19_000.0,
            order: 2,
            q: 5.0,
        }
    }

    fn biquad_real<S: Storage>(cfg: &BiquadConfig, fs: u64, st: &mut S) -> BiquadRealBlock<f32, S> {
        BiquadRealBlock::new(cfg, rate(fs), st).unwrap()
    }

    /// `fork(delay(k), biquad) -> make_complex` at 48 kHz.
    fn fork_make_complex(
        k: usize,
    ) -> (
        Fork<DelayRealBlock<f32, Vs>, BiquadRealBlock<f32, Vs>, Vs>,
        MakeComplexBlock<C32, Vs>,
    ) {
        let st = &mut VecStorage;
        let fork = Fork::new(
            delay_real(k, 48_000, st),
            biquad_real(&lowpass_cfg(), 48_000, st),
            64,
            st,
        )
        .unwrap();
        (fork, MakeComplexBlock::new(rate(48_000), st).unwrap())
    }

    /// `fork(delay(0), pilot band-pass) -> stereo_pll` at 192 kHz.
    fn fork_stereo_pll() -> (
        Fork<DelayRealBlock<f32, Vs>, BiquadRealBlock<f32, Vs>, Vs>,
        StereoPllBlock<C32, Vs>,
    ) {
        let st = &mut VecStorage;
        let fork = Fork::new(
            delay_real(0, FS_PLL, st),
            biquad_real(&pilot_cfg(), FS_PLL, st),
            64,
            st,
        )
        .unwrap();
        let pll = StereoPllBlock::new(&StereoPllConfig::default(), rate(FS_PLL), st).unwrap();
        (fork, pll)
    }

    /// A multiplex-like signal: a 1 kHz tone, a 19 kHz pilot of level 0.1 and noise.
    fn mpx(n: usize) -> Vec<f32> {
        let noise = real_signal::<f32>(n, 476, 0.0);
        (0..n)
            .map(|i| {
                let t = i as f64 / FS_PLL as f64;
                let tone = 0.4 * (std::f64::consts::TAU * 1_000.0 * t).sin();
                let pilot = 0.1 * (std::f64::consts::TAU * 19_000.0 * t).sin();
                (tone + pilot) as f32 + 0.05 * noise[i]
            })
            .collect()
    }

    /// The 20 random cut plans (input, output) of seed 469.
    fn cut_plans() -> Vec<(ChunkPlan, ChunkPlan)> {
        let mut rng = Pcg32::new(469, 32);
        (0..20)
            .map(|_| {
                let input = ChunkPlan::new(rng.next_u64(), 1 + rng.below(400) as usize);
                let output = ChunkPlan::new(rng.next_u64(), 1 + rng.below(64) as usize);
                (input, output)
            })
            .collect()
    }

    /// The chain of `mk()` equals the step-by-step computation `want` bit for bit, for five
    /// intermediate buffer sizes and the 20 cut plans.
    fn check<A, B>(mk: impl Fn() -> (A, B), input: &[f32], want: &[B::Out])
    where
        A: Block<In = f32>,
        B: Block<In = A::Out>,
        A::Out: sdrk_blocks::ChainItem,
        B::Out: StreamElem,
    {
        assert!(want.len() > 100);
        for chunk in [1, 7, 64, 256, 4096] {
            let (a, b) = mk();
            let mut c = Chain::new(a, b, chunk, &mut VecStorage).unwrap();
            let got = run_chain(&mut c, input, None, None);
            assert_eq!(got.len(), want.len(), "chunk {chunk}");
            assert!(bits(&got) == bits(want), "chunk {chunk}");
        }
        let (a, b) = mk();
        let mut c = Chain::new(a, b, 32, &mut VecStorage).unwrap();
        for (i, (pi, po)) in cut_plans().into_iter().enumerate() {
            let got = run_chain(&mut c, input, Some(pi), Some(po));
            assert_eq!(got.len(), want.len(), "plan {i}");
            assert!(bits(&got) == bits(want), "plan {i}");
            c.reset();
        }
    }

    #[test]
    fn fork_then_make_complex_equals_the_steps() {
        let input = real_signal::<f32>(4_000, 477, 0.013);
        for k in [0usize, 3] {
            // Step by step: each branch alone, the earlier one (the IIR filter, delay 0)
            // behind k zeros, then the pairs joined.
            let re = common::run_block(&mut delay_real(k, 48_000, &mut VecStorage), &input, None);
            let lp = common::run_block(
                &mut biquad_real(&lowpass_cfg(), 48_000, &mut VecStorage),
                &input,
                None,
            );
            let im: Vec<f32> = core::iter::repeat_n(0.0, k)
                .chain(lp)
                .take(re.len())
                .collect();
            let mut want = vec![C32::default(); re.len()];
            let p = MakeComplexBlock::<C32, Vs>::new(rate(48_000), &mut VecStorage)
                .unwrap()
                .process((&re, &im), &mut want)
                .unwrap();
            assert_eq!(p.produced, re.len());
            check(|| fork_make_complex(k), &input, &want);
        }
    }

    #[test]
    fn fork_then_stereo_pll_equals_the_steps() {
        let input = mpx(40_000);
        let pilot = common::run_block(
            &mut biquad_real(&pilot_cfg(), FS_PLL, &mut VecStorage),
            &input,
            None,
        );
        let mut pll = StereoPllBlock::<C32, Vs>::new(
            &StereoPllConfig::default(),
            rate(FS_PLL),
            &mut VecStorage,
        )
        .unwrap();
        let mut want = vec![C32::default(); input.len()];
        let p = pll.process((&input, &pilot), &mut want).unwrap();
        assert_eq!(p.produced, input.len());
        assert!(
            pll.pilot_locked(),
            "the loop did not lock on the test signal"
        );
        check(fork_stereo_pll, &input, &want);
        // The chain locks as the block alone does.
        let (a, b) = fork_stereo_pll();
        let mut c = Chain::new(a, b, 64, &mut VecStorage).unwrap();
        run_chain(&mut c, &input, None, None);
        assert!(c.second().pilot_locked());
    }

    /// Bytes of all stages and the buffers of fork and chain, from their `storage_bytes`.
    fn total_bytes(
        k: usize,
        fs: u64,
        bp: &BiquadConfig,
        fork_chunk: usize,
        chain_chunk: usize,
    ) -> usize {
        let a = delay_real(k, fs, &mut VecStorage);
        let b = biquad_real(bp, fs, &mut VecStorage);
        DelayRealBlock::<f32, As>::storage_bytes(&DelayConfig { samples: k }, rate(fs)).unwrap()
            + BiquadRealBlock::<f32, As>::storage_bytes(bp, rate(fs)).unwrap()
            + Fork::<_, _, Vs>::storage_bytes(&a, &b, fork_chunk).unwrap()
            + Chain::<
                Fork<DelayRealBlock<f32, As>, BiquadRealBlock<f32, As>, As>,
                MakeComplexBlock<C32, As>,
                As,
            >::storage_bytes(chain_chunk)
    }

    #[test]
    fn storage_of_fork_and_chain_is_exact() {
        for (k, fork_chunk, chain_chunk) in [(0usize, 1usize, 1usize), (3, 64, 32), (17, 5, 256)] {
            for (fs, bp, pll) in [(48_000, lowpass_cfg(), false), (FS_PLL, pilot_cfg(), true)] {
                let total = total_bytes(k, fs, &bp, fork_chunk, chain_chunk);
                // Two f32 buffers between the stages of the chain.
                assert!(total >= 2 * 4 * chain_chunk + 3 * 4 * fork_chunk + 4 * k);
                let build = |arena: &mut ArenaStorage<'_>| -> Result<(), Error> {
                    let a = DelayRealBlock::<f32, _>::new(
                        &DelayConfig { samples: k },
                        rate(fs),
                        arena,
                    )?;
                    let b = BiquadRealBlock::<f32, _>::new(&bp, rate(fs), arena)?;
                    let fork = Fork::new(a, b, fork_chunk, arena)?;
                    if pll {
                        let j = StereoPllBlock::<C32, _>::new(
                            &StereoPllConfig::default(),
                            rate(fs),
                            arena,
                        )?;
                        Chain::new(fork, j, chain_chunk, arena).map(|_| ())
                    } else {
                        let j = MakeComplexBlock::<C32, _>::new(rate(fs), arena)?;
                        Chain::new(fork, j, chain_chunk, arena).map(|_| ())
                    }
                };
                // Fresh aligned arena: exactly the sum.
                check_exact_storage(total, build);
                // Shifted by one byte: padding of at most align_of - 1 = 3 bytes per buffer on
                // top (all buffers hold f32 here, so only the first one is padded).
                let mut mem = Aligned::new(total + 64);
                let mut arena = ArenaStorage::new(mem.bytes(total + 64).split_at_mut(1).1);
                build(&mut arena).unwrap();
                assert!(
                    arena.used() > total && arena.used() <= total + 3,
                    "k {k}, chunks {fork_chunk}/{chain_chunk}: {} of {total}",
                    arena.used()
                );
            }
        }
    }

    #[test]
    fn chains_with_a_fork_are_send_and_sync() {
        assert_send_sync::<
            Chain<
                Fork<DelayRealBlock<f32, Vs>, BiquadRealBlock<f32, Vs>, Vs>,
                StereoPllBlock<C32, Vs>,
                Vs,
            >,
        >();
        assert_send_sync::<
            Chain<
                Fork<DelayRealBlock<f32, As>, BiquadRealBlock<f32, As>, As>,
                MakeComplexBlock<C32, As>,
                As,
            >,
        >();
    }
}
