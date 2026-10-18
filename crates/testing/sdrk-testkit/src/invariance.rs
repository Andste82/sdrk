//! Block-size-invariance harness.
//!
//! Every stateful kernel, block and later the pipeline must produce
//! bit-identical output no matter how the input is split into calls. The
//! harness runs a step function once over the whole input and once over
//! random chunks and compares both outputs.

use sdrk_core::Progress;

use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;
use sdrk_core::Sample;

const CASE: &str = "block_invariance";

/// Deterministic sequence of chunk sizes, uniform in `1..=max`.
///
/// ```
/// use sdrk_testkit::ChunkPlan;
/// let plan = ChunkPlan::new(1, 5);
/// let sizes: Vec<usize> = plan.sizes().take(100).collect();
/// assert!(sizes.iter().all(|&s| (1..=5).contains(&s)));
/// assert_eq!(sizes, plan.sizes().take(100).collect::<Vec<_>>());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPlan {
    seed: u64,
    max: usize,
}

impl ChunkPlan {
    /// Creates a plan; a `max` of 0 is treated as 1.
    pub fn new(seed: u64, max: usize) -> Self {
        Self {
            seed,
            max: max.max(1),
        }
    }

    /// Seed of the plan.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Largest chunk size.
    pub fn max(&self) -> usize {
        self.max
    }

    /// An endless iterator over the chunk sizes; every call starts from the
    /// beginning.
    pub fn sizes(&self) -> ChunkSizes {
        ChunkSizes {
            rng: Pcg32::new(self.seed, 0x5eed),
            max: u32::try_from(self.max).unwrap_or(u32::MAX),
        }
    }
}

/// Iterator returned by [`ChunkPlan::sizes`].
#[derive(Debug, Clone)]
pub struct ChunkSizes {
    rng: Pcg32,
    max: u32,
}

impl Iterator for ChunkSizes {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        Some(self.rng.below(self.max) as usize + 1)
    }
}

/// What a passing invariance check observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvarianceReport {
    /// Number of output samples (equal in both runs).
    pub outputs: usize,
    /// `step` calls of the whole-input run.
    pub calls_whole: usize,
    /// `step` calls of the chunked run.
    pub calls_chunked: usize,
}

/// Feeds `input` to `step` in pieces of the given sizes (`None` offers all
/// remaining input every time). Returns the produced count and the calls.
fn run_pass<I: Copy, O, S>(
    input: &[I],
    out: &mut [O],
    mut sizes: Option<ChunkSizes>,
    state: &mut S,
    step: &mut impl FnMut(&mut S, &[I], &mut [O]) -> Result<Progress, sdrk_core::Error>,
) -> Result<(usize, usize), Failure> {
    let mut pos = 0usize;
    let mut produced = 0usize;
    let mut calls = 0usize;
    while pos < input.len() {
        let rest = input.get(pos..).unwrap_or_default();
        let want = sizes
            .as_mut()
            .and_then(Iterator::next)
            .unwrap_or(rest.len());
        let (offered, _) = rest
            .split_at_checked(want.min(rest.len()))
            .unwrap_or_default();
        let room = out.get_mut(produced..).unwrap_or_default();
        let room_len = room.len();
        let p = step(state, offered, room).map_err(Failure::from)?;
        calls += 1;
        if p.consumed > offered.len() || p.produced > room_len {
            return Err(
                Failure::new(CASE, "step reported impossible progress").detail(format_args!(
                    "consumed {} of {}, produced {} of {}",
                    p.consumed,
                    offered.len(),
                    p.produced,
                    room_len
                )),
            );
        }
        if p.consumed == 0 && p.produced == 0 {
            return Err(
                Failure::new(CASE, "step made no progress").detail(format_args!(
                    "input position {pos}, offered {}",
                    offered.len()
                )),
            );
        }
        pos += p.consumed;
        produced += p.produced;
    }
    Ok((produced, calls))
}

/// Runs `step` over `input` in one piece and again in chunks from `plan`, each
/// time with a fresh state from `make_state`, and requires identical output.
///
/// `step` may consume less than it is offered; the harness resubmits the rest.
/// `out_whole` and `out_chunked` receive the outputs of the two runs and must
/// be large enough for all outputs.
///
/// # Errors
/// A [`Failure`] if `step` fails or stalls, if the output counts differ, or
/// if any output sample differs in its raw bits. The detail of a mismatch
/// starts with `first mismatch at index N`.
///
/// ```
/// use sdrk_core::Progress;
/// use sdrk_testkit::{check_block_invariance, ChunkPlan};
/// let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
/// let (mut a, mut b) = (vec![0f32; 100], vec![0f32; 100]);
/// let report = check_block_invariance(
///     &input, &mut a, &mut b, ChunkPlan::new(1, 9),
///     || (),
///     |_, i, o| {
///         let n = i.len().min(o.len());
///         o[..n].copy_from_slice(&i[..n]);
///         Ok(Progress { consumed: n, produced: n })
///     },
/// )?;
/// assert_eq!(report.outputs, 100);
/// # Ok::<(), sdrk_testkit::Failure>(())
/// ```
pub fn check_block_invariance<I: Copy, O: Sample, S>(
    input: &[I],
    out_whole: &mut [O],
    out_chunked: &mut [O],
    plan: ChunkPlan,
    mut make_state: impl FnMut() -> S,
    mut step: impl FnMut(&mut S, &[I], &mut [O]) -> Result<Progress, sdrk_core::Error>,
) -> Result<InvarianceReport, Failure> {
    let mut state = make_state();
    let (n_whole, calls_whole) = run_pass(input, out_whole, None, &mut state, &mut step)?;
    let mut state = make_state();
    let (n_chunked, calls_chunked) = run_pass(
        input,
        out_chunked,
        Some(plan.sizes()),
        &mut state,
        &mut step,
    )?;
    if n_whole != n_chunked {
        return Err(Failure::new(CASE, "output counts differ")
            .detail(format_args!("whole {n_whole}, chunked {n_chunked}")));
    }
    let whole = out_whole.get(..n_whole).unwrap_or_default();
    let chunked = out_chunked.get(..n_chunked).unwrap_or_default();
    if let Some(index) = bit_exact(whole, chunked) {
        return Err(Failure::new(CASE, "output depends on the block size")
            .detail(format_args!("first mismatch at index {index}")));
    }
    Ok(InvarianceReport {
        outputs: n_whole,
        calls_whole,
        calls_chunked,
    })
}
