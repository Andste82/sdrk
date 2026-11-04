//! `Fork` (M31 §2-§4): the cases that `tests/fork.rs` does not cover: the first branch as the
//! delayed one, validation with synthetic timings, unequal output lengths, branch errors and
//! exact storage.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::large_stack_frames,
    clippy::many_single_char_names,
    clippy::items_after_statements
)]

mod common;

use common::{check_exact_storage, rate};
use sdrk::{Delay, Progress, Ratio, RatioKind};
use sdrk_blocks::{
    ArenaStorage, Block, BlockDescription, BlockTiming, DecimateConfig, DecimateRealBlock,
    DelayConfig, DelayRealBlock, Error, FORK_DESCRIPTION, Fork, ParamValue, all_descriptions,
};
use sdrk_testkit::VecStorage;

type Vs = VecStorage;

fn delay(samples: usize) -> DelayRealBlock<f32, Vs> {
    DelayRealBlock::new(&DelayConfig { samples }, rate(48_000), &mut VecStorage).unwrap()
}

fn decimate(factor: usize, attenuation_db: f64) -> DecimateRealBlock<f32, Vs> {
    let cfg = DecimateConfig {
        factor,
        attenuation_db,
        ..DecimateConfig::default()
    };
    DecimateRealBlock::new(&cfg, rate(48_000), &mut VecStorage).unwrap()
}

/// Test-only pass-through block with a configurable timing.
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

fn timed(num: u64, den: u64, delay_num: u64, delay_den: u64, kind: RatioKind) -> Timed {
    Timed(BlockTiming {
        ratio: Ratio::new(num, den).unwrap(),
        kind,
        delay: Delay::new(delay_num, delay_den).unwrap(),
    })
}

/// Runs a fork over `input` in one call per pass; returns both output streams.
fn run_fork<A, B>(fork: &mut Fork<A, B, Vs>, input: &[f32]) -> (Vec<f32>, Vec<f32>)
where
    A: Block<In = f32, Out = f32>,
    B: Block<In = f32, Out = f32>,
{
    let total = input.len() + 16;
    let (mut x, mut y) = (vec![0.0f32; total], vec![0.0f32; total]);
    let (mut pos_in, mut pos_out, mut idle) = (0, 0, 0);
    while idle < 4 {
        let (n_in, n_out) = (input.len() - pos_in, total - pos_out);
        let p = fork
            .process(
                &input[pos_in..pos_in + n_in],
                (
                    &mut x[pos_out..pos_out + n_out],
                    &mut y[pos_out..pos_out + n_out],
                ),
            )
            .unwrap();
        pos_in += p.consumed;
        pos_out += p.produced;
        idle = if p.consumed + p.produced == 0 {
            idle + 1
        } else {
            0
        };
    }
    assert_eq!(pos_in, input.len());
    x.truncate(pos_out);
    y.truncate(pos_out);
    (x, y)
}

#[test]
fn the_first_branch_is_delayed_when_it_is_earlier() {
    let mut fork = Fork::new(delay(0), delay(4), 16, &mut VecStorage).unwrap();
    let mut input = [0.0f32; 10];
    input[1] = 1.0;
    let (x, y) = run_fork(&mut fork, &input);
    assert_eq!(x.iter().position(|v| *v == 1.0), Some(5));
    assert_eq!(y.iter().position(|v| *v == 1.0), Some(5));
    assert_eq!(fork.timing().delay, Delay::new(4, 1).unwrap());
}

#[test]
fn validation() {
    let constant = RatioKind::Constant;
    let bad = |r: Result<Fork<Timed, Timed, Vs>, Error>| match r {
        Err(Error::InvalidParameter { reason, .. }) => reason,
        other => panic!("expected InvalidParameter, got {other:?}"),
    };
    let new = |a, b, chunk| Fork::<Timed, Timed, Vs>::new(a, b, chunk, &mut VecStorage);
    bad(new(
        timed(1, 1, 0, 1, constant),
        timed(1, 1, 0, 1, constant),
        0,
    ));
    bad(new(
        timed(1, 2, 0, 1, constant),
        timed(1, 4, 0, 1, constant),
        8,
    ));
    bad(new(
        timed(1, 1, 0, 1, RatioKind::Variable),
        timed(1, 1, 0, 1, constant),
        8,
    ));
    // 15.5 input samples at ratio 1/2 are 7.75 output samples.
    let reason = bad(new(
        timed(1, 2, 31, 2, constant),
        timed(1, 2, 0, 1, constant),
        8,
    ));
    assert_eq!(
        reason,
        "fork branches are misaligned by a fraction of a sample"
    );
    // 3/2 against 1/2 input samples at ratio 1: a whole sample apart.
    let fork = new(timed(1, 1, 3, 2, constant), timed(1, 1, 1, 2, constant), 8).unwrap();
    assert_eq!(fork.timing().delay, Delay::new(3, 2).unwrap());
    // 16 input samples at ratio 1/4 are 4 output samples.
    let fork = new(timed(1, 4, 16, 1, constant), timed(1, 4, 0, 1, constant), 8).unwrap();
    assert!(format!("{fork:?}").contains("delay_b: 4"));
    // chunk below min_input of a branch.
    let a = decimate(4, 60.0);
    let min = a.min_input();
    if min > 1 {
        assert!(matches!(
            Fork::new(a, decimate(4, 60.0), min - 1, &mut VecStorage),
            Err(Error::InvalidParameter { name: "chunk", .. })
        ));
    }
}

#[test]
fn unequal_output_lengths_are_rejected() {
    let mut fork = Fork::new(delay(0), delay(0), 8, &mut VecStorage).unwrap();
    let (mut x, mut y) = ([0.0f32; 4], [0.0f32; 3]);
    assert!(matches!(
        fork.process(&[1.0; 4], (&mut x, &mut y)),
        Err(Error::Core(sdrk_core::Error::LengthMismatch { .. }))
    ));
}

#[test]
fn storage_is_exact() {
    type As<'a> = ArenaStorage<'a>;
    for (da, db, chunk) in [
        (0usize, 0usize, 1usize),
        (5, 0, 32),
        (0, 3, 17),
        (40, 1, 256),
    ] {
        let bytes = {
            let a = DelayRealBlock::<f32, Vs>::new(
                &DelayConfig { samples: da },
                rate(48_000),
                &mut VecStorage,
            )
            .unwrap();
            let b = DelayRealBlock::<f32, Vs>::new(
                &DelayConfig { samples: db },
                rate(48_000),
                &mut VecStorage,
            )
            .unwrap();
            Fork::<_, _, Vs>::storage_bytes(&a, &b, chunk).unwrap()
        };
        check_exact_storage(bytes, |arena: &mut As<'_>| {
            let a = DelayRealBlock::<f32, Vs>::new(
                &DelayConfig { samples: da },
                rate(48_000),
                &mut VecStorage,
            )?;
            let b = DelayRealBlock::<f32, Vs>::new(
                &DelayConfig { samples: db },
                rate(48_000),
                &mut VecStorage,
            )?;
            Fork::new(a, b, chunk, arena).map(|_| ())
        });
    }
}

#[test]
fn branch_errors_come_back_unchanged() {
    let mut fork = Fork::new(delay(2), delay(1), 8, &mut VecStorage).unwrap();
    assert_eq!(fork.param("1.samples").unwrap(), ParamValue::I64(1));
    assert_eq!(
        fork.set_param("0.samples", ParamValue::I64(3)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(fork.param("samples"), Err(Error::UnknownParameter));
    assert!(all_descriptions().iter().all(|d| d.kind != "fork"));
}
