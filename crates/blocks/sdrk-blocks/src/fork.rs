//! `Fork`: one input, two branches, a pair of streams out, with automatic delay alignment.

use sdrk::{Delay, Progress, RatioKind};

use crate::block::{Block, BlockTiming, ParamValue};
use crate::blocks::lockstep;
use crate::describe::BlockDescription;
use crate::error::{Error, Result};
use crate::ports::{Ports, StreamElem, Two};
use crate::storage::{Plan, Storage};

/// Description of a [`Fork`]: kind `fork`, no ports (they depend on the branches). A fork is a
/// Rust construct only: the kind is not registered in the pipeline and not part of
/// [`all_descriptions`](crate::all_descriptions).
///
/// ```
/// use sdrk_blocks::FORK_DESCRIPTION;
/// assert_eq!(FORK_DESCRIPTION.kind, "fork");
/// assert!(FORK_DESCRIPTION.inputs.is_empty() && FORK_DESCRIPTION.outputs.is_empty());
/// ```
pub static FORK_DESCRIPTION: BlockDescription = BlockDescription {
    kind: "fork",
    summary: "One input, two branches, a pair of streams out (Rust only).",
    details: "Both branches read the same input stream; their outputs leave the fork as a pair \
        of streams that advance together. The branches must have the same constant ratio; \
        the branch with the smaller delay is delayed by whole output samples so that both \
        outputs are aligned. The ports follow from the branches and cannot be described \
        statically, so inputs and outputs are empty. Parameters are addressed as \
        `<branch>.<name>` with branch 0 for the first and 1 for the second; nested chains and \
        forks add prefixes (`1.0.cutoff_hz`).",
    inputs: &[],
    outputs: &[],
    params: &[],
    required: &[],
};

/// One input, two branches: `A` and `B` read the same stream, and the output is the pair
/// `Two<A::Out, B::Out>`. A fork is a [`Block`] itself, so it can stand in a
/// [`Chain`](crate::Chain) (followed by a block that reads a pair, such as
/// [`MakeComplexBlock`](crate::MakeComplexBlock)). Forks nest only through such a join (a branch
/// can be a chain of a fork and a joining block): the output of each branch must be a stream
/// element, and the fork's own output `Two<..>` is not one, so there is no three-way fork.
///
/// Both branches must have the same ratio of kind `Constant`. Their delays, converted to output
/// samples, may differ by a whole number `k` of samples: the fork then delays the output of the
/// branch with the smaller delay by `k` samples (a ring buffer of `k` elements that starts with
/// zeros), so that both outputs are aligned. A fractional difference is rejected.
///
/// The fork keeps the input it has copied but not both branches have read, and the output of
/// each branch that has not been paired yet, in buffers of `chunk` samples from a [`Storage`].
/// `Progress::consumed` is what the fork copied into its input buffer, `Progress::produced` the
/// number of pairs. The output does not depend on how the caller cuts input and output.
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, DelayConfig, DelayRealBlock, Fork};
/// let rate = Rate::hz(48_000)?;
/// let mut mem = AlignedBytes([0u8; 4096]);
/// let mut arena = ArenaStorage::new(&mut mem.0);
/// // Branch 0 delays by 3 samples, branch 1 not at all: the fork delays branch 1 by 3.
/// let slow = DelayRealBlock::<f32, _>::new(&DelayConfig { samples: 3 }, rate, &mut arena)?;
/// let fast = DelayRealBlock::<f32, _>::new(&DelayConfig { samples: 0 }, rate, &mut arena)?;
/// let mut fork = Fork::new(slow, fast, 64, &mut arena)?;
/// let mut impulse = [0.0f32; 8];
/// impulse[0] = 1.0;
/// let (mut x, mut y) = ([0.0f32; 8], [0.0f32; 8]);
/// let p = fork.process(&impulse, (&mut x, &mut y))?;
/// assert_eq!((p.consumed, p.produced), (8, 8));
/// // The impulse appears at the same index in both outputs.
/// assert_eq!(x, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
/// assert_eq!(x, y);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Fork<A, B, S: Storage>
where
    A: Block,
    B: Block<In = A::In>,
    A::In: StreamElem,
    A::Out: StreamElem,
    B::Out: StreamElem,
{
    a: A,
    b: B,
    /// Input copied by the fork; `input[..in_fill]` is valid.
    input: S::Buf<A::In>,
    in_fill: usize,
    /// How far each branch has read `input`.
    read_a: usize,
    read_b: usize,
    /// Output of each branch that has not been paired yet; `out_x[..fill_x]` is valid.
    out_a: S::Buf<A::Out>,
    fill_a: usize,
    out_b: S::Buf<B::Out>,
    fill_b: usize,
    /// The last call of a branch filled all the room it was given, so it may still owe output.
    a_owes: bool,
    b_owes: bool,
    /// Alignment delay lines: at most one of them is not empty.
    ring_a: S::Buf<A::Out>,
    ring_b: S::Buf<B::Out>,
    /// Position in the non-empty ring.
    pos: usize,
}

impl<A, B, S: Storage> core::fmt::Debug for Fork<A, B, S>
where
    A: Block,
    B: Block<In = A::In>,
    A::In: StreamElem,
    A::Out: StreamElem,
    B::Out: StreamElem,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Fork")
            .field("in_fill", &self.in_fill)
            .field("read_a", &self.read_a)
            .field("read_b", &self.read_b)
            .field("fill_a", &self.fill_a)
            .field("fill_b", &self.fill_b)
            .field("delay_a", &self.ring_a.as_ref().len())
            .field("delay_b", &self.ring_b.as_ref().len())
            .finish_non_exhaustive()
    }
}

/// The validated shape of a fork: the alignment delays (in output samples) of each branch.
struct Layout {
    delay_a: usize,
    delay_b: usize,
}

fn invalid(name: &'static str, reason: &'static str) -> Error {
    Error::InvalidParameter { name, reason }
}

/// Checks the branches and `chunk` and computes the alignment.
fn layout<A: Block, B: Block>(a: &A, b: &B, chunk: usize) -> Result<Layout> {
    if chunk == 0 {
        return Err(invalid("chunk", "the chunk of a fork must be at least 1"));
    }
    if chunk < a.min_input() || chunk < b.min_input() {
        return Err(invalid(
            "chunk",
            "the chunk of a fork is smaller than the min_input of a branch",
        ));
    }
    let (ta, tb) = (a.timing(), b.timing());
    if ta.kind != RatioKind::Constant || tb.kind != RatioKind::Constant {
        return Err(invalid(
            "fork",
            "fork branches need a ratio of kind constant",
        ));
    }
    if ta.ratio != tb.ratio {
        return Err(invalid("fork", "fork branches have different ratios"));
    }
    // Delays in output samples: delay * ratio = delay / (1 / ratio), exact.
    let inverse = ta.ratio.inverse();
    let da = ta.delay.div_ratio(inverse)?.reduced();
    let db = tb.delay.div_ratio(inverse)?.reduced();
    let (left, right, den) = cross(da, db);
    let misaligned = || {
        invalid(
            "fork",
            "fork branches are misaligned by a fraction of a sample",
        )
    };
    let diff = left.abs_diff(right);
    if !diff.is_multiple_of(den) {
        return Err(misaligned());
    }
    let k = usize::try_from(diff / den).map_err(|_| {
        invalid(
            "fork",
            "the alignment delay of a fork does not fit the address space",
        )
    })?;
    Ok(if left > right {
        // `a` is later: delay the output of `b`.
        Layout {
            delay_a: 0,
            delay_b: k,
        }
    } else {
        Layout {
            delay_a: k,
            delay_b: 0,
        }
    })
}

/// Both delays over the common denominator `da.den * db.den`: `(da.num * db.den,
/// db.num * da.den, da.den * db.den)`, exact in `u128`.
fn cross(da: Delay, db: Delay) -> (u128, u128, u128) {
    (
        u128::from(da.num()) * u128::from(db.den()),
        u128::from(db.num()) * u128::from(da.den()),
        (u128::from(da.den()) * u128::from(db.den())).max(1),
    )
}

/// The bytes `new` takes, in the order of the allocations.
fn plan<A: Block, B: Block>(chunk: usize, layout: &Layout) -> Result<Plan>
where
    A::In: StreamElem,
    A::Out: StreamElem,
    B::Out: StreamElem,
{
    let mut plan = Plan::default();
    plan.add::<A::In>(chunk)?;
    plan.add::<A::Out>(chunk)?;
    plan.add::<B::Out>(chunk)?;
    plan.add::<A::Out>(layout.delay_a)?;
    plan.add::<B::Out>(layout.delay_b)?;
    Ok(plan)
}

impl<A, B, S: Storage> Fork<A, B, S>
where
    A: Block,
    B: Block<In = A::In>,
    A::In: StreamElem,
    A::Out: StreamElem,
    B::Out: StreamElem,
{
    /// Forks the input into `a` and `b`. `chunk` is the capacity of the input buffer and of
    /// each branch's output buffer in samples; the buffers and the alignment delay line come
    /// from `storage`.
    ///
    /// # Errors
    /// `InvalidParameter` if `chunk` is 0 or smaller than the `min_input` of a branch, if a
    /// branch has a ratio of kind `Variable`, if the ratios differ, or if the delays differ by
    /// a fraction of an output sample (`"fork branches are misaligned by a fraction of a
    /// sample"`); `Storage` if the storage cannot provide the buffers.
    ///
    /// ```
    /// use sdrk::Rate;
    /// use sdrk_blocks::{
    ///     AlignedBytes, ArenaStorage, DecimateConfig, DecimateRealBlock, DelayConfig,
    ///     DelayRealBlock, Error, Fork,
    /// };
    /// let rate = Rate::hz(48_000)?;
    /// let mut mem = AlignedBytes([0u8; 8192]);
    /// let mut arena = ArenaStorage::new(&mut mem.0);
    /// let a = DelayRealBlock::<f32, _>::new(&DelayConfig { samples: 0 }, rate, &mut arena)?;
    /// let b = DecimateRealBlock::<f32, _>::new(&DecimateConfig::default(), rate, &mut arena)?;
    /// // Ratio 1 against a decimator: the branches do not run at the same rate.
    /// assert!(matches!(Fork::new(a, b, 64, &mut arena), Err(Error::InvalidParameter { .. })));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(a: A, b: B, chunk: usize, storage: &mut S) -> Result<Self> {
        let layout = layout(&a, &b, chunk)?;
        let input = storage.alloc::<A::In>(chunk)?;
        let out_a = storage.alloc::<A::Out>(chunk)?;
        let out_b = storage.alloc::<B::Out>(chunk)?;
        let ring_a = storage.alloc::<A::Out>(layout.delay_a)?;
        let ring_b = storage.alloc::<B::Out>(layout.delay_b)?;
        Ok(Self {
            a,
            b,
            input,
            in_fill: 0,
            read_a: 0,
            read_b: 0,
            out_a,
            fill_a: 0,
            out_b,
            fill_b: 0,
            a_owes: false,
            b_owes: false,
            ring_a,
            ring_b,
            pos: 0,
        })
    }

    /// Bytes `new` takes from the storage for these branches and this `chunk`: the input
    /// buffer (`chunk` samples of `A::In`), one output buffer of `chunk` samples per branch and
    /// the alignment delay line of `k` samples of the delayed branch's output type, with the
    /// padding that aligns each buffer behind the previous one.
    ///
    /// On an arena that starts at an address aligned to [`ARENA_ALIGN`](crate::ARENA_ALIGN)
    /// (an [`AlignedBytes`](crate::AlignedBytes)) and is still empty, `new` takes exactly this.
    /// At a position that is not aligned for an element, up to `align_of - 1` bytes of padding
    /// per buffer come on top.
    ///
    /// # Errors
    /// The errors of [`Fork::new`] except `Storage`.
    ///
    /// ```
    /// use sdrk::Rate;
    /// use sdrk_blocks::{AlignedBytes, ArenaStorage, DelayConfig, DelayRealBlock, Fork};
    /// let rate = Rate::hz(48_000)?;
    /// let mut mem = AlignedBytes([0u8; 4096]);
    /// let mut arena = ArenaStorage::new(&mut mem.0);
    /// let a = DelayRealBlock::<f32, _>::new(&DelayConfig { samples: 5 }, rate, &mut arena)?;
    /// let b = DelayRealBlock::<f32, _>::new(&DelayConfig { samples: 0 }, rate, &mut arena)?;
    /// // Three buffers of 32 `f32` and a delay line of 5 `f32` for branch 1.
    /// let bytes = Fork::<_, _, ArenaStorage<'_>>::storage_bytes(&a, &b, 32)?;
    /// assert_eq!(bytes, (3 * 32 + 5) * 4);
    /// let before = arena.used();
    /// let _fork = Fork::new(a, b, 32, &mut arena)?;
    /// assert_eq!(arena.used() - before, bytes);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn storage_bytes(a: &A, b: &B, chunk: usize) -> Result<usize> {
        let layout = layout(a, b, chunk)?;
        Ok(plan::<A, B>(chunk, &layout)?.bytes())
    }

    /// The first branch (parameter prefix `0.`).
    pub fn first(&self) -> &A {
        &self.a
    }

    /// The first branch, mutable.
    pub fn first_mut(&mut self) -> &mut A {
        &mut self.a
    }

    /// The second branch (parameter prefix `1.`).
    pub fn second(&self) -> &B {
        &self.b
    }

    /// The second branch, mutable.
    pub fn second_mut(&mut self) -> &mut B {
        &mut self.b
    }

    /// Moves as many pairs as both branch buffers hold (at most the length of the outputs) to
    /// the front of `out_a` and `out_b`, through the alignment delay line; returns their number.
    fn pairs_out(&mut self, out_a: &mut [A::Out], out_b: &mut [B::Out]) -> Result<usize> {
        let n = self
            .fill_a
            .min(self.fill_b)
            .min(out_a.len())
            .min(out_b.len());
        if n == 0 {
            return Ok(0);
        }
        let dst_a = out_a.get_mut(..n).ok_or_else(internal)?;
        let src_a = self.out_a.as_ref().get(..n).ok_or_else(internal)?;
        emit(src_a, dst_a, self.ring_a.as_mut(), &mut self.pos)?;
        let dst_b = out_b.get_mut(..n).ok_or_else(internal)?;
        let src_b = self.out_b.as_ref().get(..n).ok_or_else(internal)?;
        emit(src_b, dst_b, self.ring_b.as_mut(), &mut self.pos)?;
        move_front(self.out_a.as_mut(), n, self.fill_a)?;
        move_front(self.out_b.as_mut(), n, self.fill_b)?;
        self.fill_a -= n;
        self.fill_b -= n;
        Ok(n)
    }

    /// Drops the input both branches have read and copies as much of `input` as fits behind
    /// the rest; returns the number of samples copied.
    fn refill(&mut self, input: &[A::In]) -> Result<usize> {
        let drop = self.read_a.min(self.read_b);
        if drop > 0 {
            move_front(self.input.as_mut(), drop, self.in_fill)?;
            self.in_fill -= drop;
            self.read_a -= drop;
            self.read_b -= drop;
        }
        let free = self
            .input
            .as_mut()
            .get_mut(self.in_fill..)
            .ok_or_else(internal)?;
        let n = free.len().min(input.len());
        let dst = free.get_mut(..n).ok_or_else(internal)?;
        let src = input.get(..n).ok_or_else(internal)?;
        dst.copy_from_slice(src);
        self.in_fill += n;
        Ok(n)
    }

    /// Calls each branch once (if it has room and something to do); returns whether either
    /// consumed or produced anything.
    fn run_branches(&mut self) -> Result<bool> {
        let buf = self
            .input
            .as_ref()
            .get(..self.in_fill)
            .ok_or_else(internal)?;
        let ran_a = run_branch(
            &mut self.a,
            buf,
            &mut self.read_a,
            self.out_a.as_mut(),
            &mut self.fill_a,
            &mut self.a_owes,
        )?;
        let ran_b = run_branch(
            &mut self.b,
            buf,
            &mut self.read_b,
            self.out_b.as_mut(),
            &mut self.fill_b,
            &mut self.b_owes,
        )?;
        Ok(ran_a || ran_b)
    }

    /// Whether branch `a` has the larger (or the same) delay.
    fn a_is_later(&self) -> bool {
        let (ta, tb) = (self.a.timing(), self.b.timing());
        let (left, right, _) = cross(ta.delay, tb.delay);
        left >= right
    }
}

fn internal() -> Error {
    Error::Core(sdrk_core::Error::Internal("fork buffer"))
}

/// `s` without its first `n` samples (empty if `n` is beyond the length).
fn skip<T>(s: &mut [T], n: usize) -> &mut [T] {
    let n = n.min(s.len());
    s.split_at_mut_checked(n)
        .map(|(_, rest)| rest)
        .unwrap_or_default()
}

/// Moves `s[start..end]` to the front of `s`.
fn move_front<T: Copy>(s: &mut [T], start: usize, end: usize) -> Result<()> {
    if start > end || end > s.len() {
        return Err(internal());
    }
    s.copy_within(start..end, 0);
    Ok(())
}

/// Writes `src` through the delay line `ring` (position `pos`) into `dst`: each output is the
/// sample `ring.len()` steps earlier. An empty ring copies. `dst` and `src` have equal lengths.
fn emit<T: Copy>(src: &[T], dst: &mut [T], ring: &mut [T], pos: &mut usize) -> Result<()> {
    if src.len() != dst.len() {
        return Err(internal());
    }
    if ring.is_empty() {
        dst.copy_from_slice(src);
        return Ok(());
    }
    let (mut src, mut dst) = (src, dst);
    while !src.is_empty() {
        let k = ring.len();
        let seg = src.len().min(k.saturating_sub(*pos));
        let held = ring.get_mut(*pos..*pos + seg).ok_or_else(internal)?;
        let (s, s_rest) = src.split_at_checked(seg).ok_or_else(internal)?;
        let (d, d_rest) = core::mem::take(&mut dst)
            .split_at_mut_checked(seg)
            .ok_or_else(internal)?;
        d.copy_from_slice(held);
        held.copy_from_slice(s);
        (src, dst) = (s_rest, d_rest);
        *pos += seg;
        if *pos >= k {
            *pos = 0;
        }
    }
    Ok(())
}

/// Calls `block` once on its unread input `input[*read..fill_in]` with the free part of its
/// output buffer, if it has room and either unread input or output it may still owe. Returns
/// whether the block consumed or produced anything.
fn run_branch<K: Block>(
    block: &mut K,
    input: &[K::In],
    read: &mut usize,
    out: &mut [K::Out],
    fill: &mut usize,
    owes: &mut bool,
) -> Result<bool>
where
    K::In: StreamElem,
    K::Out: StreamElem,
{
    let unread = input.get(*read..).ok_or_else(internal)?;
    let free = out.get_mut(*fill..).ok_or_else(internal)?;
    let room = free.len();
    if room == 0 || (unread.is_empty() && !*owes) {
        return Ok(false);
    }
    let p = block.process(K::In::view(unread), K::Out::view_mut(free))?;
    *owes = p.produced >= room;
    *read += p.consumed.min(unread.len());
    *fill += p.produced.min(room);
    Ok(p.consumed > 0 || p.produced > 0)
}

impl<A, B, S: Storage> Block for Fork<A, B, S>
where
    A: Block,
    B: Block<In = A::In>,
    A::In: StreamElem,
    A::Out: StreamElem,
    B::Out: StreamElem,
{
    type In = A::In;
    type Out = Two<A::Out, B::Out>;

    fn description(&self) -> &'static BlockDescription {
        &FORK_DESCRIPTION
    }

    fn process(
        &mut self,
        input: <A::In as Ports>::Ref<'_>,
        (out_a, out_b): (&mut [A::Out], &mut [B::Out]),
    ) -> Result<Progress> {
        lockstep(out_a, out_b)?;
        let (mut input, mut out_a, mut out_b) = (A::In::slice(input), out_a, out_b);
        let mut total = Progress::default();
        loop {
            // 1. Move pairs out while both branch buffers hold samples.
            let n = self.pairs_out(out_a, out_b)?;
            out_a = skip(out_a, n);
            out_b = skip(out_b, n);
            total.produced += n;
            if out_a.is_empty() {
                break;
            }
            // 2. Drop the input both branches have read, copy new input behind the rest.
            let copied = self.refill(input)?;
            input = input.get(copied..).unwrap_or_default();
            total.consumed += copied;
            // 3. Run each branch on its unread part.
            let ran = self.run_branches()?;
            if n == 0 && copied == 0 && !ran {
                break;
            }
        }
        Ok(total)
    }

    fn timing(&self) -> BlockTiming {
        let (ta, tb) = (self.a.timing(), self.b.timing());
        let kind = if ta.kind != RatioKind::Constant || tb.kind != RatioKind::Constant {
            RatioKind::Variable
        } else {
            RatioKind::Constant
        };
        let delay = if self.a_is_later() {
            ta.delay
        } else {
            tb.delay
        };
        BlockTiming {
            ratio: ta.ratio,
            kind,
            delay,
        }
    }

    fn reset(&mut self) {
        self.a.reset();
        self.b.reset();
        self.in_fill = 0;
        self.read_a = 0;
        self.read_b = 0;
        self.fill_a = 0;
        self.fill_b = 0;
        self.a_owes = false;
        self.b_owes = false;
        self.ring_a.as_mut().fill(Default::default());
        self.ring_b.as_mut().fill(Default::default());
        self.pos = 0;
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name.split_once('.') {
            Some(("0", rest)) => self.a.set_param(rest, value),
            Some(("1", rest)) => self.b.set_param(rest, value),
            _ => Err(Error::UnknownParameter),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        match name.split_once('.') {
            Some(("0", rest)) => self.a.param(rest),
            Some(("1", rest)) => self.b.param(rest),
            _ => Err(Error::UnknownParameter),
        }
    }

    fn min_input(&self) -> usize {
        self.a.min_input().max(self.b.min_input())
    }

    fn map_index(&self, rel_in: u64) -> u64 {
        if self.a_is_later() {
            self.a.map_index(rel_in)
        } else {
            self.b.map_index(rel_in)
        }
    }
}
