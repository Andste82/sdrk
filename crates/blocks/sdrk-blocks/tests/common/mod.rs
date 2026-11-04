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
use sdrk_blocks::{
    ArenaStorage, Block, DesignSpec, Error, Frame, FrameMeta, FrameOut, StreamElem, TunedElem,
};
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
pub fn run_block<I: StreamElem, O: StreamElem + Default, B: Block<In = I, Out = O>>(
    block: &mut B,
    input: &[I],
    plan: Option<ChunkPlan>,
) -> Vec<O> {
    let mut out = Vec::new();
    let mut room = vec![O::default(); 4096];
    let mut pending: Vec<I> = Vec::new();
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
pub fn run_frames<I: StreamElem, B: Block<In = I, Out = Frame>>(
    block: &mut B,
    input: &[I],
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
            let mut meta = None;
            let out_port = FrameOut {
                values: &mut frame,
                meta: &mut meta,
            };
            let used = block.process(piece, out_port).unwrap().consumed;
            if let Some(meta) = meta {
                out.push((frame.clone(), meta));
            }
            assert!(used > 0, "process made no progress");
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

/// 20 chunk plans for one block: sizes from 1 to 700, the seed of the block plus the index.
pub fn twenty_plans(seed: u64) -> Vec<ChunkPlan> {
    let max = [
        1, 2, 3, 5, 7, 11, 16, 33, 64, 100, 128, 255, 256, 257, 300, 512, 700, 1000, 1500, 4000,
    ];
    max.iter()
        .enumerate()
        .map(|(i, m)| ChunkPlan::new(seed * 100 + i as u64, *m))
        .collect()
}

/// `a` and `b` are equal element for element; the panic names the first difference.
pub fn same_bits<T: PartialEq + core::fmt::Debug>(a: &[T], b: &[T], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths differ");
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        panic!("{what}: first difference at {i}: {:?} vs {:?}", a[i], b[i]);
    }
}

/// Whole-buffer output equals `want`, and every chunk plan gives the same bits.
pub fn check_block<B, F>(want: &[B::Out], input: &[B::In], seed: u64, make: F, what: &str)
where
    B: Block,
    B::Out: PartialEq + core::fmt::Debug + sdrk_blocks::StreamElem,
    B::In: sdrk_blocks::StreamElem,
    F: Fn() -> B,
{
    check_block_with(want, input, seed, make, what, true);
}

/// As `check_block`; with `exact_chunks` false the chunked runs of `f32` or `C32` output only
/// need an SNR of 120 dB against `want` (for kernels whose `f32` bits on a SIMD level depend on
/// the split, such as the `f32` `deemphasis`, see docs/kernels.md).
pub fn check_block_with<B, F>(
    want: &[B::Out],
    input: &[B::In],
    seed: u64,
    make: F,
    what: &str,
    exact_chunks: bool,
) where
    B: Block,
    B::Out: PartialEq + core::fmt::Debug + sdrk_blocks::StreamElem,
    B::In: sdrk_blocks::StreamElem,
    F: Fn() -> B,
{
    let whole = run_block(&mut make(), input, None);
    same_bits(want, &whole, &format!("{what} whole"));
    for plan in twenty_plans(seed) {
        let chunked = run_block(&mut make(), input, Some(plan));
        if exact_chunks {
            same_bits(want, &chunked, &format!("{what} {plan:?}"));
        } else {
            let bytes_w: &[u8] = bytemuck::cast_slice(want);
            let bytes_c: &[u8] = bytemuck::cast_slice(&chunked);
            if bytes_w != bytes_c {
                let w: &[f32] = bytemuck::cast_slice(want);
                let c: &[f32] = bytemuck::cast_slice(&chunked);
                assert_eq!(w.len(), c.len(), "{what} {plan:?}");
                let sig: f64 = w.iter().map(|x| f64::from(*x).powi(2)).sum();
                let err: f64 = w
                    .iter()
                    .zip(c)
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                    .sum();
                assert!(10.0 * (sig / err).log10() >= 120.0, "{what} {plan:?}");
            }
        }
    }
}

/// The taps of `spec` at the rate `fs` (at most 1023), scaled and converted as the tuned
/// resampler converts them.
pub fn taps_for<T: TunedElem>(spec: &DesignSpec, fs: f64, scale: f64) -> Vec<T::Tap> {
    let mut design = vec![0.0f64; spec.num_taps(fs, 1023).unwrap()];
    spec.design(fs, 1023, &mut design).unwrap();
    design
        .iter()
        .map(|v| <T as TunedElem>::tap(*v * scale).unwrap())
        .collect()
}
