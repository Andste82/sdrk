//! Helpers shared by the integration tests of `sdrk-blocks`.
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_lossless,
    clippy::panic,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::assert_is_empty,
    clippy::range_plus_one,
    clippy::large_stack_arrays,
    clippy::doc_markdown,
    clippy::multiple_bound_locations,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_pass_by_value,
    clippy::type_complexity,
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation
)]

use num_complex::Complex64;
use sdrk::{ComplexSample, Rate};
use sdrk_blocks::{ArenaStorage, Block, Error, FrameBlock, FrameMeta};
use sdrk_testkit::{ChunkPlan, Pcg32};

pub fn rate(hz: u64) -> Rate {
    Rate::hz(hz).unwrap()
}

/// Deterministic test signal: a tone plus noise, amplitude about 0.5.
pub fn signal<T: ComplexSample>(n: usize, seed: u64, tone_cycles_per_sample: f64) -> Vec<T> {
    let mut rng = Pcg32::new(seed, 7);
    (0..n)
        .map(|i| {
            let phase = std::f64::consts::TAU * tone_cycles_per_sample * i as f64;
            let noise = Complex64::new(rng.normal(), rng.normal()) * 0.02;
            T::from_c64_round(Complex64::new(0.4 * phase.cos(), 0.4 * phase.sin()) + noise)
        })
        .collect()
}

/// A real test signal: a tone plus noise.
pub fn real_signal<R: sdrk::RealSample>(n: usize, seed: u64, cycles_per_sample: f64) -> Vec<R> {
    let mut rng = Pcg32::new(seed, 9);
    (0..n)
        .map(|i| {
            let phase = std::f64::consts::TAU * cycles_per_sample * i as f64;
            R::from_f64_round(0.4 * phase.sin() + 0.02 * rng.normal())
        })
        .collect()
}

/// Feeds `input` to the block in pieces from `plan` (all at once for `None`) and resubmits
/// what the block did not consume together with the next piece.
pub fn run_block<B: Block>(block: &mut B, input: &[B::In], plan: Option<ChunkPlan>) -> Vec<B::Out>
where
    B::Out: Default,
{
    let mut out = Vec::new();
    let mut room = vec![B::Out::default(); 4096];
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
            if (p.consumed == 0 && p.produced == 0)
                || (pending.is_empty() && p.produced < room.len())
            {
                break;
            }
        }
    }
    out
}

/// Pushes `input` into a frame block in pieces from `plan` and collects the frames.
pub fn run_frames<B: FrameBlock>(
    block: &mut B,
    input: &[B::In],
    plan: Option<ChunkPlan>,
) -> Vec<(Vec<f32>, FrameMeta)> {
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

/// The chunk plans used for the invariance checks.
pub fn plans(seed: u64) -> [ChunkPlan; 3] {
    [
        ChunkPlan::new(seed, 1),
        ChunkPlan::new(seed + 1, 7),
        ChunkPlan::new(seed + 2, 700),
    ]
}

/// 16-byte aligned memory for arenas.
pub struct Aligned(Vec<u128>);

impl Aligned {
    pub fn new(bytes: usize) -> Self {
        Self(vec![0u128; bytes.div_ceil(16) + 1])
    }

    pub fn bytes(&mut self, len: usize) -> &mut [u8] {
        &mut bytemuck::cast_slice_mut::<u128, u8>(&mut self.0)[..len]
    }
}

/// Builds with an arena of exactly `bytes` bytes (16-byte aligned) and requires that the arena
/// is used up completely; with one byte less the build must fail with a `Storage` error.
pub fn check_exact_storage<F>(bytes: usize, build: F)
where
    F: for<'a> Fn(&mut ArenaStorage<'a>) -> Result<(), Error>,
{
    let mut mem = Aligned::new(bytes + 32);
    let mut arena = ArenaStorage::new(mem.bytes(bytes));
    build(&mut arena).unwrap();
    assert_eq!(arena.remaining(), 0, "storage_bytes is not exact");
    assert_eq!(arena.used(), bytes);
    if bytes > 0 {
        let mut mem = Aligned::new(bytes + 32);
        let mut arena = ArenaStorage::new(mem.bytes(bytes - 1));
        match build(&mut arena) {
            Err(Error::Storage { needed, available }) => {
                assert!(needed > available, "needed {needed}, available {available}");
            }
            other => panic!("expected a storage error, got {other:?}"),
        }
    }
}

/// Phase angle of a complex number.
pub fn arg(c: Complex64) -> f64 {
    c.im.atan2(c.re)
}

/// Magnitude of a complex number.
pub fn norm(c: Complex64) -> f64 {
    c.re.hypot(c.im)
}
