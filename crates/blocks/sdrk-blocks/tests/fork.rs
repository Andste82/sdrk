//! `Fork` with real blocks (M31 tests 1, 2, 3 and 5): equality with separate runs of the
//! branches for random cut plans, delay alignment, rejected ratios and chunk sizes, parameter
//! paths through chains and forks, `ParamMap`, block-size invariance and reset.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::large_stack_frames,
    clippy::type_complexity,
    clippy::too_many_lines,
    clippy::assert_is_empty
)]

mod common;

use common::{rate, real_signal, run_block, signal};
use sdrk::{C32, Delay, Progress, Ratio, RatioKind};
use sdrk_blocks::design::BiquadResponse;
use sdrk_blocks::{
    BiquadConfig, BiquadRealBlock, Block, BlockDescription, BlockTiming, Chain, DecimateConfig,
    DecimateRealBlock, DelayBlock, DelayConfig, DelayRealBlock, Error, FORK_DESCRIPTION, FirBlock,
    FirConfig, Fork, ParamMap, ParamValue, ResampleConfig, ResampleRealBlock, RotateBlock,
    RotateConfig, StreamElem, Two,
};
use sdrk_testkit::{ChunkPlan, Pcg32, VecStorage};

type Vs = VecStorage;

const FS: u64 = 48_000;
const MISALIGNED: &str = "fork branches are misaligned by a fraction of a sample";

// ---------------------------------------------------------------------------------------
// blocks

fn delay_real(samples: usize) -> DelayRealBlock<f32, Vs> {
    DelayRealBlock::new(&DelayConfig { samples }, rate(FS), &mut VecStorage).unwrap()
}

fn delay_complex(samples: usize) -> DelayBlock<C32, Vs> {
    DelayBlock::new(&DelayConfig { samples }, rate(FS), &mut VecStorage).unwrap()
}

fn biquad_real() -> BiquadRealBlock<f32, Vs> {
    let cfg = BiquadConfig {
        response: BiquadResponse::Lowpass,
        cutoff_hz: 3_000.0,
        order: 4,
        ..BiquadConfig::default()
    };
    BiquadRealBlock::new(&cfg, rate(FS), &mut VecStorage).unwrap()
}

fn decimate_cfg(factor: usize, attenuation_db: f64) -> DecimateConfig {
    DecimateConfig {
        factor,
        attenuation_db,
        ..DecimateConfig::default()
    }
}

fn decimate(cfg: &DecimateConfig) -> DecimateRealBlock<f32, Vs> {
    DecimateRealBlock::new(cfg, rate(FS), &mut VecStorage).unwrap()
}

/// A decimator by `factor` with exactly `taps` taps (the transition width is searched).
fn decimate_taps(factor: usize, taps: usize) -> DecimateRealBlock<f32, Vs> {
    let cfg = (1..2_000)
        .map(|i| DecimateConfig {
            factor,
            cutoff_hz: Some(4_000.0),
            transition_hz: Some(f64::from(i) * 10.0),
            ..DecimateConfig::default()
        })
        .find(|c| c.num_taps(rate(FS)).ok() == Some(taps))
        .unwrap_or_else(|| panic!("no decimator by {factor} with {taps} taps"));
    decimate(&cfg)
}

/// A complex FIR low-pass with exactly `taps` taps.
fn fir_taps(taps: usize) -> FirBlock<C32, Vs> {
    let cfg = (1..2_000)
        .map(|i| FirConfig {
            cutoff_hz: 6_000.0,
            transition_hz: f64::from(i) * 10.0,
            ..FirConfig::default()
        })
        .find(|c| c.num_taps(rate(FS)).ok() == Some(taps))
        .unwrap_or_else(|| panic!("no fir with {taps} taps"));
    FirBlock::new(&cfg, rate(FS), &mut VecStorage).unwrap()
}

fn rotate(frequency_hz: f64) -> RotateBlock<C32, Vs> {
    RotateBlock::new(&RotateConfig { frequency_hz }, rate(FS), &mut VecStorage).unwrap()
}

/// Test-only pass-through block with a given timing, for timings no real block has.
struct Timed(BlockTiming);

impl Block for Timed {
    type In = f32;
    type Out = f32;
    fn description(&self) -> &'static BlockDescription {
        &FORK_DESCRIPTION
    }
    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<Progress, Error> {
        let n = input.len().min(output.len());
        output[..n].copy_from_slice(&input[..n]);
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }
    fn timing(&self) -> BlockTiming {
        self.0
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

// ---------------------------------------------------------------------------------------
// runners

fn bits<T: StreamElem>(v: &[T]) -> &[u8] {
    bytemuck::cast_slice(v)
}

/// The 20 random cut plans (input, output) of the tests, from seed 469.
fn cut_plans() -> Vec<(ChunkPlan, ChunkPlan)> {
    let mut rng = Pcg32::new(469, 31);
    (0..20)
        .map(|_| {
            let input = ChunkPlan::new(rng.next_u64(), 1 + rng.below(400) as usize);
            let output = ChunkPlan::new(rng.next_u64(), 1 + rng.below(64) as usize);
            (input, output)
        })
        .collect()
}

/// Runs a block with a pair output over `input`, with input pieces and output sizes from the
/// plans (everything at once for `None`), until it neither consumes nor produces anything.
fn run_pair<K, I, X, Y>(
    block: &mut K,
    input: &[I],
    plans: Option<&(ChunkPlan, ChunkPlan)>,
) -> (Vec<X>, Vec<Y>)
where
    K: Block<In = I, Out = Two<X, Y>>,
    I: StreamElem,
    X: StreamElem,
    Y: StreamElem,
{
    let total = input.len() + 64;
    let (mut x, mut y) = (vec![X::default(); total], vec![Y::default(); total]);
    let mut ins = plans.map(|(i, _)| i.sizes());
    let mut outs = plans.map(|(_, o)| o.sizes());
    let (mut pos_in, mut pos_out, mut idle) = (0, 0, 0);
    while idle < 3 {
        let n_in = ins
            .as_mut()
            .map_or(input.len(), |s| s.next().unwrap())
            .min(input.len() - pos_in);
        let n_out = outs
            .as_mut()
            .map_or(total, |s| s.next().unwrap())
            .min(total - pos_out);
        let p = block
            .process(
                I::view(&input[pos_in..pos_in + n_in]),
                (
                    &mut x[pos_out..pos_out + n_out],
                    &mut y[pos_out..pos_out + n_out],
                ),
            )
            .unwrap();
        assert!(p.consumed <= n_in && p.produced <= n_out);
        pos_in += p.consumed;
        pos_out += p.produced;
        idle = if p.consumed + p.produced == 0 {
            idle + 1
        } else {
            0
        };
    }
    assert_eq!(pos_in, input.len(), "the fork did not take all input");
    x.truncate(pos_out);
    y.truncate(pos_out);
    (x, y)
}

/// What a fork must produce: each branch run alone on the whole input, the branch with the
/// smaller delay behind `k` zeros (`delay_x` and `delay_y` in output samples), as many pairs as
/// both branches produced (the delayed branch keeps its last `k` samples in the delay line).
fn expected<X: StreamElem, Y: StreamElem>(
    alone_x: &[X],
    alone_y: &[Y],
    delay_x: usize,
    delay_y: usize,
) -> (Vec<X>, Vec<Y>) {
    let n = alone_x.len().min(alone_y.len());
    let mut x: Vec<X> = core::iter::repeat_n(X::default(), delay_x)
        .chain(alone_x.iter().copied())
        .collect();
    let mut y: Vec<Y> = core::iter::repeat_n(Y::default(), delay_y)
        .chain(alone_y.iter().copied())
        .collect();
    x.truncate(n);
    y.truncate(n);
    (x, y)
}

/// The fork of `mk()` equals the separate runs of its branches bit for bit: for five chunk
/// sizes without cuts (block-size invariance) and for the 20 cut plans of seed 469, and after a
/// reset in between.
fn check_fork<A, B, I>(mk: impl Fn() -> (A, B), input: &[I], delays: (usize, usize))
where
    A: Block<In = I>,
    B: Block<In = I>,
    I: StreamElem,
    A::Out: StreamElem,
    B::Out: StreamElem,
{
    let (mut a, mut b) = mk();
    let min = a.min_input().max(b.min_input());
    let alone_a = run_block(&mut a, input, None);
    let alone_b = run_block(&mut b, input, None);
    let (want_a, want_b) = expected(&alone_a, &alone_b, delays.0, delays.1);
    assert!(want_a.len() > 100, "too little output to compare");
    for chunk in [min, min + 6, 64, 256, 4096] {
        let (a, b) = mk();
        let mut fork = Fork::new(a, b, chunk, &mut VecStorage).unwrap();
        let (x, y) = run_pair(&mut fork, input, None);
        assert_eq!(x.len(), want_a.len(), "chunk {chunk}: number of pairs");
        assert!(bits(&x) == bits(&want_a), "chunk {chunk}: first branch");
        assert!(bits(&y) == bits(&want_b), "chunk {chunk}: second branch");
    }
    let (a, b) = mk();
    let mut fork = Fork::new(a, b, 128, &mut VecStorage).unwrap();
    for (i, plans) in cut_plans().iter().enumerate() {
        let (x, y) = run_pair(&mut fork, input, Some(plans));
        assert_eq!(x.len(), want_a.len(), "plan {i}: number of pairs");
        assert!(bits(&x) == bits(&want_a), "plan {i}: first branch");
        assert!(bits(&y) == bits(&want_b), "plan {i}: second branch");
        fork.reset();
    }
}

fn reason<T: core::fmt::Debug>(r: Result<T, Error>) -> &'static str {
    match r {
        Err(Error::InvalidParameter { reason, .. }) => reason,
        other => panic!("expected InvalidParameter, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------
// test 1: fork equals separate runs

#[test]
fn delay_and_biquad_real_equal_separate_runs() {
    let input = real_signal::<f32>(5_000, 469, 0.031);
    // An IIR stage reports a delay of 0: no alignment.
    assert_eq!(biquad_real().timing().delay, Delay::ZERO);
    check_fork(|| (delay_real(0), biquad_real()), &input, (0, 0));
}

#[test]
fn two_decimators_by_4_with_different_filters_equal_separate_runs() {
    let input = real_signal::<f32>(12_000, 470, 0.017);
    let (ca, cb) = (decimate_cfg(4, 60.0), decimate_cfg(4, 80.0));
    let r = Ratio::new(1, 4).unwrap();
    let out_delay = |d: &DecimateRealBlock<f32, Vs>| {
        d.timing()
            .delay
            .div_ratio(r.inverse())
            .unwrap()
            .reduced()
            .as_f64()
    };
    let (da, db) = (out_delay(&decimate(&ca)), out_delay(&decimate(&cb)));
    assert_ne!(da, db, "the filters must differ in length");
    let k = (da - db).abs();
    assert_eq!(k.fract(), 0.0, "the test needs a whole-sample difference");
    let delays = if da > db {
        (0, k as usize)
    } else {
        (k as usize, 0)
    };
    check_fork(|| (decimate(&ca), decimate(&cb)), &input, delays);
}

#[test]
fn two_rotators_equal_separate_runs() {
    let input = signal::<C32>(5_000, 471, 0.01);
    check_fork(|| (rotate(1_500.0), rotate(-7_250.0)), &input, (0, 0));
}

// ---------------------------------------------------------------------------------------
// test 2: alignment

#[test]
fn a_31_tap_fir_and_delay_0_are_aligned_by_15_samples() {
    let fir = fir_taps(31);
    assert_eq!(fir.timing().delay.reduced(), Delay::new(15, 1).unwrap());

    // An impulse appears at the same index in both outputs.
    let mut impulse = vec![C32::default(); 400];
    impulse[40] = C32::new(1.0, 0.0);
    let mut fork = Fork::new(fir_taps(31), delay_complex(0), 64, &mut VecStorage).unwrap();
    assert!(format!("{fork:?}").contains("delay_b: 15"), "{fork:?}");
    assert_eq!(fork.timing().delay.reduced(), Delay::new(15, 1).unwrap());
    let (x, y) = run_pair(&mut fork, &impulse, None);
    assert_eq!(x.len(), impulse.len());
    let peak = |v: &[C32]| {
        (0..v.len())
            .max_by(|&i, &j| v[i].norm_sqr().total_cmp(&v[j].norm_sqr()))
            .unwrap()
    };
    assert_eq!(peak(&x), 55);
    assert_eq!(peak(&y), 55);
    assert_eq!(y[55], C32::new(1.0, 0.0));

    // 15 zeros inserted on the short branch, then the input unchanged.
    let input = signal::<C32>(3_000, 472, 0.02);
    for (i, plans) in cut_plans().iter().enumerate() {
        let mut fork = Fork::new(fir_taps(31), delay_complex(0), 32, &mut VecStorage).unwrap();
        let (x, y) = run_pair(&mut fork, &input, Some(plans));
        assert_eq!(y.len(), input.len(), "plan {i}");
        assert!(y[..15].iter().all(|v| *v == C32::default()), "plan {i}");
        assert!(
            bits(&y[15..]) == bits(&input[..input.len() - 15]),
            "plan {i}"
        );
        let alone = run_block(&mut fir_taps(31), &input, None);
        assert!(bits(&x) == bits(&alone), "plan {i}");
    }
}

#[test]
fn a_fractional_delay_difference_is_rejected() {
    // The block design always gives an odd number of taps, so a 31-tap decimator by 2 stands
    // for the misalignment: 15 input samples are 7.5 output samples (D352).
    let d31 = decimate_taps(2, 31);
    assert_eq!(d31.timing().delay.reduced(), Delay::new(15, 1).unwrap());
    assert_eq!(
        reason(Fork::new(d31, decimate_taps(2, 33), 64, &mut VecStorage)),
        MISALIGNED
    );
    // 15 and 17 input samples are 7.5 and 8.5 output samples: one whole sample apart.
    let fork = Fork::new(
        decimate_taps(2, 31),
        decimate_taps(2, 35),
        64,
        &mut VecStorage,
    )
    .unwrap();
    assert!(format!("{fork:?}").contains("delay_a: 1"), "{fork:?}");
    // The numbers of the specification: 32 taps are 15.5 input samples, 7.75 output samples.
    let timed = |delay_num, delay_den| {
        Timed(BlockTiming {
            ratio: Ratio::new(1, 2).unwrap(),
            kind: RatioKind::Constant,
            delay: Delay::new(delay_num, delay_den).unwrap(),
        })
    };
    assert_eq!(
        reason(Fork::<_, _, Vs>::new(
            timed(31, 2),
            timed(0, 1),
            8,
            &mut VecStorage
        )),
        MISALIGNED
    );
    // The literal case of the specification, a decimator by 2 against delay(0), is rejected
    // earlier, because the ratios differ (D352).
    assert_eq!(
        reason(Fork::new(
            decimate_taps(2, 31),
            delay_real(0),
            64,
            &mut VecStorage,
        )),
        "fork branches have different ratios"
    );
    // The rejections do not depend on the order of the branches.
    assert_eq!(
        reason(Fork::new(
            decimate_taps(2, 33),
            decimate_taps(2, 31),
            64,
            &mut VecStorage
        )),
        MISALIGNED
    );
}

// ---------------------------------------------------------------------------------------
// test 3: rate mismatch, Variable ratios, chunk sizes

#[test]
fn ratios_1_2_and_1_4_are_rejected() {
    let (d2, d4) = (
        decimate(&decimate_cfg(2, 60.0)),
        decimate(&decimate_cfg(4, 60.0)),
    );
    assert_eq!(d2.timing().ratio, Ratio::new(1, 2).unwrap());
    assert_eq!(d4.timing().ratio, Ratio::new(1, 4).unwrap());
    reason(Fork::new(d2, d4, 64, &mut VecStorage));
    let (d2, d4) = (
        decimate(&decimate_cfg(2, 60.0)),
        decimate(&decimate_cfg(4, 60.0)),
    );
    reason(Fork::new(d4, d2, 64, &mut VecStorage));
    reason(Fork::<_, _, Vs>::storage_bytes(
        &decimate(&decimate_cfg(2, 60.0)),
        &decimate(&decimate_cfg(4, 60.0)),
        64,
    ));
}

#[test]
fn a_variable_ratio_is_rejected() {
    let tuned = || {
        let cfg = ResampleConfig {
            rate_out_hz: 24_000.0,
            tunable: true,
            ..ResampleConfig::default()
        };
        ResampleRealBlock::<f32, Vs>::new(&cfg, rate(FS), &mut VecStorage).unwrap()
    };
    let r = tuned();
    assert_eq!(r.timing().kind, RatioKind::Variable);
    assert_eq!(r.timing().ratio, Ratio::new(1, 2).unwrap());
    let d2 = || decimate(&decimate_cfg(2, 60.0));
    reason(Fork::new(r, d2(), 64, &mut VecStorage));
    reason(Fork::new(d2(), tuned(), 64, &mut VecStorage));
    reason(Fork::<_, _, Vs>::storage_bytes(&d2(), &tuned(), 64));
}

#[test]
fn chunk_0_and_a_chunk_below_min_input_are_rejected() {
    let d4 = || decimate(&decimate_cfg(4, 60.0));
    assert_eq!(d4().min_input(), 4);
    for chunk in [0, 1, 3] {
        assert!(
            matches!(
                Fork::new(d4(), d4(), chunk, &mut VecStorage),
                Err(Error::InvalidParameter { name: "chunk", .. })
            ),
            "chunk {chunk}"
        );
        assert!(Fork::<_, _, Vs>::storage_bytes(&d4(), &d4(), chunk).is_err());
    }
    // The larger min_input of the two branches counts.
    assert!(Fork::new(delay_real(0), biquad_real(), 0, &mut VecStorage).is_err());
    let fork = Fork::new(d4(), d4(), 4, &mut VecStorage).unwrap();
    assert_eq!(fork.min_input(), 4);
    let fork = Fork::new(delay_real(0), biquad_real(), 1, &mut VecStorage).unwrap();
    assert_eq!(fork.min_input(), 1);
}

#[test]
fn timing_of_a_decimating_fork() {
    let (ca, cb) = (decimate_cfg(4, 60.0), decimate_cfg(4, 80.0));
    let (a, b) = (decimate(&ca), decimate(&cb));
    let (ta, tb) = (a.timing(), b.timing());
    assert_ne!(ta.delay, tb.delay);
    let (want_delay, want_index) = if ta.delay.as_f64() > tb.delay.as_f64() {
        (ta.delay, a.map_index(1_000))
    } else {
        (tb.delay, b.map_index(1_000))
    };
    let fork = Fork::new(a, b, 64, &mut VecStorage).unwrap();
    let t = fork.timing();
    assert_eq!(t.ratio, Ratio::new(1, 4).unwrap());
    assert_eq!(t.kind, RatioKind::Constant);
    assert_eq!(t.delay, want_delay);
    assert_eq!(fork.map_index(1_000), want_index);
}

// ---------------------------------------------------------------------------------------
// test 5: parameters

/// `rotate -> fork(rotate, rotate)`: paths `0.frequency_hz`, `1.0.frequency_hz` and
/// `1.1.frequency_hz`.
fn shifter() -> Chain<RotateBlock<C32, Vs>, Fork<RotateBlock<C32, Vs>, RotateBlock<C32, Vs>, Vs>, Vs>
{
    let fork = Fork::new(rotate(0.0), rotate(0.0), 64, &mut VecStorage).unwrap();
    Chain::new(rotate(0.0), fork, 64, &mut VecStorage).unwrap()
}

/// The output of `shifter()` with the three frequencies set through `set`, computed with the
/// stages one after the other.
fn shifter_alone(input: &[C32], f: [f64; 3]) -> (Vec<C32>, Vec<C32>) {
    let set = |f: f64| {
        let mut r = rotate(0.0);
        r.set_param("frequency_hz", ParamValue::F64(f)).unwrap();
        r
    };
    let mid = run_block(&mut set(f[0]), input, None);
    (
        run_block(&mut set(f[1]), &mid, None),
        run_block(&mut set(f[2]), &mid, None),
    )
}

#[test]
fn parameter_paths_through_chain_and_fork_reach_the_right_block() {
    let input = signal::<C32>(2_000, 473, 0.005);
    let mut c = shifter();
    c.set_param("1.0.frequency_hz", ParamValue::F64(2_000.0))
        .unwrap();
    c.set_param("1.1.frequency_hz", ParamValue::F64(-3_000.0))
        .unwrap();
    c.set_param("0.frequency_hz", ParamValue::F64(500.0))
        .unwrap();
    assert_eq!(
        c.param("1.0.frequency_hz").unwrap(),
        ParamValue::F64(2_000.0)
    );
    assert_eq!(
        c.param("1.1.frequency_hz").unwrap(),
        ParamValue::F64(-3_000.0)
    );
    assert_eq!(c.param("0.frequency_hz").unwrap(), ParamValue::F64(500.0));
    assert_eq!(
        c.second().first().param("frequency_hz").unwrap(),
        ParamValue::F64(2_000.0)
    );
    assert_eq!(
        c.second().second().param("frequency_hz").unwrap(),
        ParamValue::F64(-3_000.0)
    );
    let (x, y) = run_pair(&mut c, &input, None);
    let (want_x, want_y) = shifter_alone(&input, [500.0, 2_000.0, -3_000.0]);
    assert_eq!(bits(&x), bits(&want_x));
    assert_eq!(bits(&y), bits(&want_y));

    // Wrong paths.
    for name in [
        "frequency_hz",
        "1.frequency_hz",
        "1.2.frequency_hz",
        "2.frequency_hz",
        "1.0.cutoff_hz",
        "1.0",
        "1.",
        "",
    ] {
        assert_eq!(
            c.set_param(name, ParamValue::F64(1.0)),
            Err(Error::UnknownParameter),
            "{name:?}"
        );
        assert_eq!(c.param(name), Err(Error::UnknownParameter), "{name:?}");
    }
    // The error of the inner block comes back unchanged.
    assert!(matches!(
        c.set_param("1.1.frequency_hz", ParamValue::F64(30_000.0)),
        Err(Error::OutOfRange { .. })
    ));
}

#[test]
fn a_param_map_of_flat_names_reaches_the_right_block() {
    const MAP: ParamMap = ParamMap::new(&[
        ("shift_hz", "0.frequency_hz"),
        ("lo_a_hz", "1.0.frequency_hz"),
        ("lo_b_hz", "1.1.frequency_hz"),
        ("broken", "1.2.frequency_hz"),
    ]);
    let input = signal::<C32>(2_000, 474, 0.005);
    let mut c = shifter();
    MAP.set(&mut c, "lo_b_hz", ParamValue::F64(4_000.0))
        .unwrap();
    MAP.set(&mut c, "shift_hz", ParamValue::F64(-1_000.0))
        .unwrap();
    assert_eq!(MAP.get(&c, "lo_b_hz").unwrap(), ParamValue::F64(4_000.0));
    assert_eq!(MAP.get(&c, "lo_a_hz").unwrap(), ParamValue::F64(0.0));
    assert_eq!(
        c.param("1.1.frequency_hz").unwrap(),
        ParamValue::F64(4_000.0)
    );
    let (x, y) = run_pair(&mut c, &input, None);
    let (want_x, want_y) = shifter_alone(&input, [-1_000.0, 0.0, 4_000.0]);
    assert_eq!(bits(&x), bits(&want_x));
    assert_eq!(bits(&y), bits(&want_y));

    // Names that are not in the map, paths given as flat names, and a path that leads nowhere.
    for name in ["frequency_hz", "1.0.frequency_hz", "lo_c_hz", ""] {
        assert_eq!(MAP.path(name), None);
        assert_eq!(
            MAP.set(&mut c, name, ParamValue::F64(1.0)),
            Err(Error::UnknownParameter),
            "{name:?}"
        );
        assert_eq!(MAP.get(&c, name), Err(Error::UnknownParameter), "{name:?}");
    }
    assert_eq!(MAP.path("broken"), Some("1.2.frequency_hz"));
    assert_eq!(MAP.get(&c, "broken"), Err(Error::UnknownParameter));
    // The composite keeps its own description: the map does not change it.
    assert_eq!(c.description().kind, "chain");
}

// ---------------------------------------------------------------------------------------
// reset, description, Send

#[test]
fn reset_clears_the_branches_the_buffers_and_the_delay_line() {
    let input = signal::<C32>(1_500, 475, 0.02);
    let mut fork = Fork::new(fir_taps(31), delay_complex(0), 16, &mut VecStorage).unwrap();
    let plans = cut_plans();
    let first = run_pair(&mut fork, &input, Some(&plans[0]));
    // Stop in the middle of a stream, with samples in every buffer and the delay line.
    let mut x = [C32::default(); 7];
    let mut y = [C32::default(); 7];
    fork.process(&input[..300], (&mut x, &mut y)).unwrap();
    fork.reset();
    let second = run_pair(&mut fork, &input, Some(&plans[1]));
    assert_eq!(bits(&first.0), bits(&second.0));
    assert_eq!(bits(&first.1), bits(&second.1));
}

#[test]
fn a_fork_reports_the_fork_description() {
    let fork = Fork::new(rotate(0.0), rotate(1.0), 8, &mut VecStorage).unwrap();
    assert!(core::ptr::eq(
        fork.description(),
        core::ptr::addr_of!(FORK_DESCRIPTION)
    ));
    assert_eq!(fork.description().kind, "fork");
    assert!(fork.description().params.is_empty());
}

#[test]
fn fork_and_param_map_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Fork<RotateBlock<C32, Vs>, DelayBlock<C32, Vs>, Vs>>();
    assert_send_sync::<Fork<DelayRealBlock<f32, Vs>, BiquadRealBlock<f32, Vs>, Vs>>();
    assert_send_sync::<
        Fork<
            DecimateRealBlock<f32, sdrk_blocks::ArenaStorage<'static>>,
            DelayRealBlock<f32, sdrk_blocks::ArenaStorage<'static>>,
            sdrk_blocks::ArenaStorage<'static>,
        >,
    >();
    assert_send_sync::<ParamMap>();
}
