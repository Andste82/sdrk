//! `Chain` with a pair of streams (`Two`) between its stages (M31 §1): lockstep buffers,
//! independence of the cut, nesting through `chain!` and exact storage sizes.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::large_stack_frames
)]

mod common;

use common::{Aligned, check_exact_storage, rate, signal};
use sdrk::{C32, Delay, Progress, Ratio, RatioKind};
use sdrk_blocks::{
    ArenaStorage, Block, BlockDescription, BlockTiming, CHAIN_DESCRIPTION, Chain, ChainItem, Error,
    MakeComplexBlock, ParamValue, Ports, ScaleBlock, ScaleConfig, StreamElem, Two, chain,
};
use sdrk_testkit::{ChunkPlan, VecStorage};

type Vs = VecStorage;

fn unit_timing() -> BlockTiming {
    BlockTiming {
        ratio: Ratio::ONE,
        kind: RatioKind::Constant,
        delay: Delay::ZERO,
    }
}

/// Test-only block: `C32` to `Two<f32, f32>` (re, im).
struct Split;

impl Block for Split {
    type In = C32;
    type Out = Two<f32, f32>;
    fn description(&self) -> &'static BlockDescription {
        &CHAIN_DESCRIPTION
    }
    fn process(
        &mut self,
        input: &[C32],
        (re, im): (&mut [f32], &mut [f32]),
    ) -> Result<Progress, Error> {
        if re.len() != im.len() {
            return Err(Error::InvalidParameter {
                name: "output",
                reason: "length mismatch",
            });
        }
        let n = input.len().min(re.len());
        for ((x, r), i) in input.iter().zip(re.iter_mut()).zip(im.iter_mut()) {
            (*r, *i) = (x.re, x.im);
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }
    fn timing(&self) -> BlockTiming {
        unit_timing()
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

/// Test-only block: `C32` to `Two<u8, f32>` (a running counter, re), so the two buffers of the
/// chain have different element sizes and need padding between them.
struct Tag(u8);

impl Block for Tag {
    type In = C32;
    type Out = Two<u8, f32>;
    fn description(&self) -> &'static BlockDescription {
        &CHAIN_DESCRIPTION
    }
    fn process(
        &mut self,
        input: &[C32],
        (t, re): (&mut [u8], &mut [f32]),
    ) -> Result<Progress, Error> {
        let n = input.len().min(t.len()).min(re.len());
        for ((x, t), r) in input.iter().zip(t.iter_mut()).zip(re.iter_mut()) {
            *t = self.0;
            *r = x.re;
            self.0 = self.0.wrapping_add(1);
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }
    fn timing(&self) -> BlockTiming {
        unit_timing()
    }
    fn reset(&mut self) {
        self.0 = 0;
    }
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

/// Test-only block: `Two<u8, f32>` to `f32` (`re + tag`), consuming whole groups of three
/// pairs only, so the chain has to keep a remainder of the pair buffers and move it to the
/// front in lockstep.
struct Gather3;

impl Block for Gather3 {
    type In = Two<u8, f32>;
    type Out = f32;
    fn description(&self) -> &'static BlockDescription {
        &CHAIN_DESCRIPTION
    }
    fn process(&mut self, (t, re): (&[u8], &[f32]), out: &mut [f32]) -> Result<Progress, Error> {
        if t.len() != re.len() {
            return Err(Error::InvalidParameter {
                name: "input",
                reason: "length mismatch",
            });
        }
        let n = t.len().min(out.len()) / 3 * 3;
        for ((o, &t), &r) in out.iter_mut().zip(t).zip(re).take(n) {
            *o = r + f32::from(t);
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }
    fn timing(&self) -> BlockTiming {
        unit_timing()
    }
    fn reset(&mut self) {}
    fn set_param(&mut self, _: &str, _: ParamValue) -> Result<(), Error> {
        Err(Error::UnknownParameter)
    }
    fn param(&self, _: &str) -> Result<ParamValue, Error> {
        Err(Error::UnknownParameter)
    }
    fn min_input(&self) -> usize {
        3
    }
}

/// Runs a stream-to-stream block with optional input and output cut plans.
fn run<B, I, O>(block: &mut B, input: &[I], plans: Option<(u64, u64)>, total_out: usize) -> Vec<O>
where
    I: StreamElem,
    O: StreamElem,
    B: Block<In = I, Out = O>,
{
    let mut out = vec![O::default(); total_out];
    let mut ins = plans.map(|(i, _)| ChunkPlan::new(i, 300).sizes());
    let mut outs = plans.map(|(_, o)| ChunkPlan::new(o, 50).sizes());
    let (mut pos_in, mut pos_out) = (0, 0);
    let mut idle = 0;
    while pos_out < total_out && idle < 4 {
        let n_in = ins
            .as_mut()
            .map_or(input.len(), |s| s.next().unwrap())
            .min(input.len() - pos_in);
        let n_out = outs
            .as_mut()
            .map_or(total_out, |s| s.next().unwrap())
            .min(total_out - pos_out);
        let p = block
            .process(
                &input[pos_in..pos_in + n_in],
                &mut out[pos_out..pos_out + n_out],
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
    out.truncate(pos_out);
    out
}

fn bits<T: StreamElem>(v: &[T]) -> &[u8] {
    bytemuck::cast_slice(v)
}

fn make_complex() -> MakeComplexBlock<C32, Vs> {
    MakeComplexBlock::new(rate(48_000), &mut VecStorage).unwrap()
}

fn scale() -> ScaleBlock<C32, Vs> {
    ScaleBlock::new(
        &ScaleConfig { gain_db: -6.0 },
        rate(48_000),
        &mut VecStorage,
    )
    .unwrap()
}

#[test]
fn a_pair_intermediate_is_passed_in_lockstep_for_every_cut() {
    let input = signal::<C32>(3_000, 469, 0.01);
    for chunk in [1, 7, 256, 4096] {
        let mut c = Chain::new(Split, make_complex(), chunk, &mut VecStorage).unwrap();
        let got = run(&mut c, &input, None, input.len());
        assert_eq!(bits(&got), bits(&input), "chunk {chunk}");
    }
    for seed in 0..20u64 {
        let mut c = Chain::new(Split, make_complex(), 64, &mut VecStorage).unwrap();
        let got = run(&mut c, &input, Some((100 + seed, 200 + seed)), input.len());
        assert_eq!(bits(&got), bits(&input), "plan {seed}");
    }
}

#[test]
fn a_remainder_of_the_pair_buffers_moves_to_the_front_together() {
    let input = signal::<C32>(2_000, 7, 0.02);
    let want: Vec<f32> = input
        .iter()
        .enumerate()
        .take(input.len() / 3 * 3)
        .map(|(i, x)| x.re + f32::from(i as u8))
        .collect();
    for chunk in [1, 4, 5, 64] {
        let mut c = Chain::new(Tag(0), Gather3, chunk, &mut VecStorage).unwrap();
        let got = run(&mut c, &input, None, want.len());
        assert_eq!(bits(&got), bits(&want), "chunk {chunk}");
    }
    for seed in 0..20u64 {
        let mut c = Chain::new(Tag(0), Gather3, 16, &mut VecStorage).unwrap();
        let got = run(&mut c, &input, Some((300 + seed, 400 + seed)), want.len());
        assert_eq!(bits(&got), bits(&want), "plan {seed}");
        c.reset();
        let again = run(&mut c, &input, Some((500 + seed, 600 + seed)), want.len());
        assert_eq!(bits(&again), bits(&want), "after reset, plan {seed}");
    }
}

#[test]
fn chain_macro_nests_a_pair_intermediate_and_routes_parameters() {
    fn build() -> Result<impl Block<In = C32, Out = C32>, Error> {
        Ok(chain!(&mut VecStorage, 32; scale(), Split, make_complex(), scale()))
    }
    let input = signal::<C32>(1_000, 11, 0.03);
    let once = run(&mut scale(), &input, None, input.len());
    let want = run(&mut scale(), &once, None, input.len());
    let mut c = build().unwrap();
    let got = run(&mut c, &input, Some((1, 2)), input.len());
    assert_eq!(bits(&got), bits(&want));
    // stage 1.1.1 is the last scale, behind the pair
    let mut c = Chain::new(
        scale(),
        Chain::new(
            Split,
            Chain::new(make_complex(), scale(), 32, &mut VecStorage).unwrap(),
            32,
            &mut VecStorage,
        )
        .unwrap(),
        32,
        &mut VecStorage,
    )
    .unwrap();
    c.set_param("1.1.1.gain_db", ParamValue::F64(-12.0))
        .unwrap();
    assert_eq!(c.param("1.1.1.gain_db").unwrap(), ParamValue::F64(-12.0));
    assert_eq!(c.param("0.gain_db").unwrap(), ParamValue::F64(-6.0));
    assert_eq!(c.timing(), unit_timing());
    assert_eq!(c.min_input(), 1);
}

#[test]
fn storage_bytes_counts_both_buffers_of_a_pair_exactly() {
    for chunk in [1usize, 3, 7, 64, 255] {
        // u8 then f32: chunk + padding to 4 + 4 * chunk
        let want = Chain::<Tag, Gather3, ArenaStorage<'_>>::storage_bytes(chunk);
        assert_eq!(want, chunk.next_multiple_of(4) + 4 * chunk, "chunk {chunk}");
        if chunk >= 3 {
            check_exact_storage(want, |arena| {
                Chain::new(Tag(0), Gather3, chunk, arena).map(|_| ())
            });
        }
        // f32 and f32: no padding
        let want = Chain::<Split, MakeComplexBlock<C32, ArenaStorage<'_>>, ArenaStorage<'_>>::storage_bytes(chunk);
        assert_eq!(want, 2 * 4 * chunk);
        check_exact_storage(want, |arena| {
            let join =
                MakeComplexBlock::<C32, _>::new(rate(48_000), &mut ArenaStorage::new(&mut []))?;
            Chain::new(Split, join, chunk, arena).map(|_| ())
        });
        // a shifted arena needs at most the alignment of each buffer on top
        let mut mem = Aligned::new(4096);
        let mut arena = ArenaStorage::new(mem.bytes(4096).split_at_mut(1).1);
        let join =
            MakeComplexBlock::<C32, _>::new(rate(48_000), &mut ArenaStorage::new(&mut [])).unwrap();
        Chain::new(Split, join, chunk, &mut arena).unwrap();
        assert_eq!(arena.used(), want + 3, "shifted arena, chunk {chunk}");
    }
}

#[test]
fn pairs_of_stream_elements_are_chain_items() {
    fn item<T: ChainItem>() -> usize {
        <T as Ports>::COUNT
    }
    assert_eq!(item::<C32>(), 1);
    assert_eq!(item::<Two<f32, C32>>(), 2);
    assert_eq!(item::<Two<u8, sdrk::Q15>>(), 2);
}

#[test]
fn chains_with_a_pair_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Chain<Split, MakeComplexBlock<C32, Vs>, Vs>>();
    assert_send_sync::<Chain<Tag, Gather3, ArenaStorage<'static>>>();
}
