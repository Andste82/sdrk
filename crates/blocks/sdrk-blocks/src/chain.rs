//! `Chain`: two blocks in a row with an intermediate buffer (two for a pair) from caller
//! storage.

use sdrk::{Progress, Ratio, RatioKind};
use sdrk_core::timing::ceil_div;

use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::BlockDescription;
use crate::error::{Error, Result};
use crate::ports::{ChainItem, Ports, sealed::ChainBufs};
use crate::storage::Storage;

/// Description of a [`Chain`]: kind `chain`, no ports (they depend on the stages). A chain is a
/// Rust construct only: the kind is not registered in the pipeline and not part of
/// [`all_descriptions`](crate::all_descriptions).
///
/// ```
/// use sdrk_blocks::CHAIN_DESCRIPTION;
/// assert_eq!(CHAIN_DESCRIPTION.kind, "chain");
/// assert!(CHAIN_DESCRIPTION.inputs.is_empty() && CHAIN_DESCRIPTION.outputs.is_empty());
/// ```
pub static CHAIN_DESCRIPTION: BlockDescription = BlockDescription {
    kind: "chain",
    summary: "Two blocks in a row (Rust only).",
    details: "The output of the first block feeds the second through one intermediate buffer \
        from caller storage. The ports follow from the two stages and cannot be described \
        statically, so inputs and outputs are empty. Parameters are addressed as \
        `<stage>.<name>` with stage 0 for the first and 1 for the second block; nested chains \
        add prefixes (`1.0.cutoff_hz`).",
    inputs: &[],
    outputs: &[],
    params: &[],
    required: &[],
};

/// Two blocks in a row: the output of `A` feeds `B` through a buffer of `chunk` samples taken
/// from a [`Storage`]. A chain is a [`Block`] itself, so chains nest (the [`chain!`](crate::chain)
/// macro writes longer ones).
///
/// `A::Out` must be a [`ChainItem`]: a stream element ([`StreamElem`](crate::StreamElem)) or a
/// pair of streams ([`Two`](crate::Two)). For a pair the chain holds two buffers of `chunk`
/// samples that are written and consumed in lockstep. Frames (and any other port type that is
/// not a stream element) cannot pass between the stages and are rejected at compile time.
/// `A::In` and `B::Out` may be any port type, so a chain can end in a frame block.
///
/// `Progress::consumed` is what `A` consumed, `Progress::produced` what `B` produced. The
/// output does not depend on how the caller cuts input and output.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{
///     AlignedBytes, ArenaStorage, Block, Chain, ConvertBlock, ConvertConfig, DdcBlock, DdcConfig,
/// };
/// let rate = Rate::hz(48_000)?;
/// let ddc_cfg = DdcConfig { decimation: 4, max_chunk: 64, ..DdcConfig::default() };
/// let mut mem = AlignedBytes([0u8; 8192]);
/// let mut arena = ArenaStorage::new(&mut mem.0);
/// let convert = ConvertBlock::<C32, _>::new(&ConvertConfig::default(), rate, &mut arena)?;
/// let ddc = DdcBlock::<C32, _>::new(&ddc_cfg, rate, &mut arena)?;
/// let mut chain = Chain::new(convert, ddc, 64, &mut arena)?;
/// let bytes = [128u8; 256]; // 128 cu8 samples
/// let mut out = [C32::default(); 32];
/// let p = chain.process(&bytes, &mut out)?;
/// assert_eq!((p.consumed, p.produced), (256, 32));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// A pair between the stages: a block that writes `Two<f32, f32>` (here a small one written
/// for the example) feeds [`MakeComplexBlock`](crate::MakeComplexBlock).
///
/// ```
/// use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, C32};
/// use sdrk_blocks::{
///     AlignedBytes, ArenaStorage, Block, BlockDescription, BlockTiming, Chain, Error,
///     MakeComplexBlock, ParamValue, Two, CHAIN_DESCRIPTION,
/// };
///
/// /// Writes each input sample to both streams, the second one negated.
/// struct Mirror;
/// impl Block for Mirror {
///     type In = f32;
///     type Out = Two<f32, f32>;
///     fn description(&self) -> &'static BlockDescription { &CHAIN_DESCRIPTION }
///     fn process(
///         &mut self,
///         input: &[f32],
///         (re, im): (&mut [f32], &mut [f32]),
///     ) -> Result<Progress, Error> {
///         let mut n = 0;
///         for ((x, r), i) in input.iter().zip(re.iter_mut()).zip(im.iter_mut()) {
///             (*r, *i) = (*x, -*x);
///             n += 1;
///         }
///         Ok(Progress { consumed: n, produced: n })
///     }
///     fn timing(&self) -> BlockTiming {
///         BlockTiming {
///             ratio: Ratio::ONE,
///             kind: RatioKind::Constant,
///             delay: Delay::ZERO,
///         }
///     }
///     fn reset(&mut self) {}
///     fn set_param(&mut self, _: &str, _: ParamValue) -> Result<(), Error> {
///         Err(Error::UnknownParameter)
///     }
///     fn param(&self, _: &str) -> Result<ParamValue, Error> { Err(Error::UnknownParameter) }
///     fn min_input(&self) -> usize { 1 }
/// }
///
/// let rate = Rate::hz(48_000)?;
/// let mut mem = AlignedBytes([0u8; 1024]);
/// let mut arena = ArenaStorage::new(&mut mem.0);
/// let join = MakeComplexBlock::<C32, _>::new(rate, &mut arena)?;
/// // Two buffers of 16 `f32` samples.
/// type S = ArenaStorage<'static>;
/// assert_eq!(Chain::<Mirror, MakeComplexBlock<C32, S>, S>::storage_bytes(16), 2 * 16 * 4);
/// let before = arena.used();
/// let mut chain = Chain::new(Mirror, join, 16, &mut arena)?;
/// assert_eq!(arena.used() - before, 2 * 16 * 4);
/// let input = [1.0f32, 2.0, 3.0];
/// let mut out = [C32::default(); 3];
/// let p = chain.process(&input, &mut out)?;
/// assert_eq!((p.consumed, p.produced), (3, 3));
/// assert_eq!(out[2], C32::new(3.0, -3.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Chain<A, B, S: Storage>
where
    A: Block,
    B: Block<In = A::Out>,
    A::Out: ChainItem,
{
    a: A,
    b: B,
    buf: <A::Out as ChainBufs>::Bufs<S>,
    /// End of the valid samples in `buf`.
    fill: usize,
    /// Start of the samples `B` has not consumed yet.
    start: usize,
    /// The last call of `a` filled all the room it was given, so it may still owe output.
    a_full: bool,
}

impl<A, B, S: Storage> core::fmt::Debug for Chain<A, B, S>
where
    A: Block,
    B: Block<In = A::Out>,
    A::Out: ChainItem,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Chain")
            .field("fill", &self.fill)
            .field("start", &self.start)
            .field("a_full", &self.a_full)
            .finish_non_exhaustive()
    }
}

impl<A, B, S: Storage> Chain<A, B, S>
where
    A: Block,
    B: Block<In = A::Out>,
    A::Out: ChainItem,
{
    /// Joins `a` and `b`. `chunk` is the capacity of the intermediate buffer in samples of
    /// `A::Out` (for a pair: of each of the two buffers); it is raised to at least
    /// `b.min_input()` and at least 1 (a smaller buffer could never hold what `b` needs). The
    /// buffers come from `storage`.
    ///
    /// # Errors
    /// `Storage` if the storage cannot provide the buffer; an error if the combined timing
    /// does not fit the exact arithmetic.
    ///
    /// ```
    /// use sdrk::{Rate, C32};
    /// use sdrk_blocks::{
    ///     AlignedBytes, ArenaStorage, Chain, ConvertBlock, ConvertConfig, DdcBlock, DdcConfig, Error,
    /// };
    /// let rate = Rate::hz(48_000)?;
    /// let mut mem = AlignedBytes([0u8; 8192]);
    /// let mut arena = ArenaStorage::new(&mut mem.0);
    /// let a = ConvertBlock::<C32, _>::new(&ConvertConfig::default(), rate, &mut arena)?;
    /// let b = DdcBlock::<C32, _>::new(&DdcConfig { max_chunk: 64, ..DdcConfig::default() }, rate, &mut arena)?;
    /// // 1000 samples of `C32` are 8000 bytes; the arena has less left.
    /// assert!(matches!(Chain::new(a, b, 1000, &mut arena), Err(Error::Storage { .. })));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(a: A, b: B, chunk: usize, storage: &mut S) -> Result<Self> {
        let chunk = chunk.max(b.min_input()).max(1);
        // Reject timing that does not fit the exact arithmetic now, not in `timing()`.
        combined_timing(&a, &b)?;
        let buf = <A::Out as ChainBufs>::alloc(storage, chunk)?;
        Ok(Self {
            a,
            b,
            buf,
            fill: 0,
            start: 0,
            a_full: false,
        })
    }

    /// Bytes `new` takes from the storage for this `chunk`: `chunk * size_of::<A::Out>()` for
    /// a stream; for a pair `Two<X, Y>` both buffers, `chunk * size_of::<X>()` and
    /// `chunk * size_of::<Y>()`, plus the padding that aligns the second buffer behind the
    /// first (saturating).
    ///
    /// On an arena that starts at an address aligned to [`ARENA_ALIGN`](crate::ARENA_ALIGN)
    /// (an [`AlignedBytes`](crate::AlignedBytes)) and is still empty, `new` takes exactly this.
    /// At a position that is not aligned for the element, up to `align_of - 1` bytes of
    /// padding per buffer come on top. `chunk` must be at least `b.min_input()` and 1,
    /// otherwise `new` raises it and takes more.
    ///
    /// ```
    /// use sdrk::{CQ15, C32};
    /// use sdrk_blocks::{ArenaStorage, Chain, ConvertBlock, DdcBlock};
    /// type A = ConvertBlock<C32, ArenaStorage<'static>>;
    /// type B = DdcBlock<C32, ArenaStorage<'static>>;
    /// assert_eq!(Chain::<A, B, ArenaStorage<'static>>::storage_bytes(256), 256 * 8);
    /// # let _ = CQ15::default();
    /// ```
    ///
    /// The example of [`Chain`] with a pair between its stages shows the size of two buffers.
    pub fn storage_bytes(chunk: usize) -> usize {
        <A::Out as ChainBufs>::bytes(chunk)
    }

    /// The first stage.
    pub fn first(&self) -> &A {
        &self.a
    }

    /// The first stage, mutable.
    pub fn first_mut(&mut self) -> &mut A {
        &mut self.a
    }

    /// The second stage.
    pub fn second(&self) -> &B {
        &self.b
    }

    /// The second stage, mutable.
    pub fn second_mut(&mut self) -> &mut B {
        &mut self.b
    }
}

/// Ratio, kind and delay of two stages in a row.
fn combined_timing<A: Block, B: Block>(a: &A, b: &B) -> Result<BlockTiming> {
    let (ta, tb) = (a.timing(), b.timing());
    let ratio = ta.ratio.mul(tb.ratio)?;
    let kind = if ta.kind == RatioKind::Variable || tb.kind == RatioKind::Variable {
        RatioKind::Variable
    } else {
        RatioKind::Constant
    };
    let delay = ta.delay.add(tb.delay.div_ratio(ta.ratio)?)?;
    Ok(BlockTiming { ratio, kind, delay })
}

impl<A, B, S: Storage> Block for Chain<A, B, S>
where
    A: Block,
    B: Block<In = A::Out>,
    A::Out: ChainItem,
{
    type In = A::In;
    type Out = B::Out;

    fn description(&self) -> &'static BlockDescription {
        &CHAIN_DESCRIPTION
    }

    fn process(
        &mut self,
        input: <A::In as Ports>::Ref<'_>,
        output: <B::Out as Ports>::Mut<'_>,
    ) -> Result<Progress> {
        let internal = || Error::Core(sdrk_core::Error::Internal("chain buffer"));
        let min_b = self.b.min_input().max(1);
        let mut input = input;
        let mut output = output;
        let mut total = Progress::default();
        let capacity = <A::Out as ChainBufs>::capacity::<S>(&self.buf);
        loop {
            let mut moved = false;
            // 1. Feed B from the buffer.
            loop {
                let avail = self.fill.checked_sub(self.start).ok_or_else(internal)?;
                let exhausted = <A::In as Ports>::ref_len(&input) == 0;
                if avail < min_b && !exhausted {
                    break;
                }
                let pending = <A::Out as ChainBufs>::view::<S>(&self.buf, self.start, self.fill)
                    .ok_or_else(internal)?;
                let p = self
                    .b
                    .process(pending, <B::Out as Ports>::mut_reborrow(&mut output))?;
                if p.consumed == 0 && p.produced == 0 {
                    break;
                }
                moved = true;
                self.start += p.consumed.min(avail);
                total.produced += p.produced;
                output = <B::Out as Ports>::mut_skip(output, p.produced);
                if <B::Out as Ports>::mut_len(&output) == 0 {
                    break;
                }
            }
            if <B::Out as Ports>::mut_len(&output) == 0 {
                break;
            }
            // 2. Move what B has not consumed to the front.
            if self.start > 0 {
                if self.fill > capacity {
                    return Err(internal());
                }
                <A::Out as ChainBufs>::move_to_front::<S>(&mut self.buf, self.start, self.fill)
                    .ok_or_else(internal)?;
                self.fill -= self.start;
                self.start = 0;
            }
            // 3. Feed the buffer from A.
            let room = capacity.checked_sub(self.fill).ok_or_else(internal)?;
            if room > 0 && (<A::In as Ports>::ref_len(&input) > 0 || self.a_full) {
                let free = <A::Out as ChainBufs>::view_mut_from::<S>(&mut self.buf, self.fill)
                    .ok_or_else(internal)?;
                let p = self.a.process(input, free)?;
                self.a_full = p.produced >= room;
                if p.consumed > 0 || p.produced > 0 {
                    moved = true;
                    input = <A::In as Ports>::ref_skip(input, p.consumed);
                    total.consumed += p.consumed;
                    self.fill += p.produced.min(room);
                }
            }
            if !moved {
                break;
            }
        }
        Ok(total)
    }

    fn timing(&self) -> BlockTiming {
        combined_timing(&self.a, &self.b).unwrap_or_else(|_| {
            // Only reachable when a run-time parameter change pushed a stage past the exact
            // arithmetic after `new` had checked it; report the first stage's timing.
            self.a.timing()
        })
    }

    fn reset(&mut self) {
        self.a.reset();
        self.b.reset();
        self.fill = 0;
        self.start = 0;
        self.a_full = false;
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
        let ratio: Ratio = self.a.timing().ratio;
        let need = ceil_div(
            u128::try_from(self.b.min_input())
                .unwrap_or(u128::MAX)
                .saturating_mul(u128::from(ratio.den())),
            u128::from(ratio.num()),
        )
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(usize::MAX);
        need.max(self.a.min_input())
    }

    fn map_index(&self, rel_in: u64) -> u64 {
        self.b.map_index(self.a.map_index(rel_in))
    }

    fn frame_len(&self) -> usize {
        self.b.frame_len()
    }
}

/// Builds a [`Chain`] of two to eight blocks, nested to the right:
/// `chain!(storage, chunk; a, b, c)` is `Chain::new(a, Chain::new(b, c, chunk, storage)?,
/// chunk, storage)?`, so the parameter names read `0.`, `1.0.` and `1.1.`.
///
/// `storage` is an expression of type `&mut S` (it is used once per joint, so write
/// `&mut arena` or a variable that is a `&mut` reference); `chunk` the capacity of every
/// intermediate buffer in samples. The macro uses `?`: the caller returns a `Result` whose
/// error type can be made from [`Error`](crate::Error).
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{
///     chain, AlignedBytes, ArenaStorage, Block, ConvertBlock, ConvertConfig, DdcBlock, DdcConfig,
///     FmDemodBlock, FmDemodConfig, ParamValue,
/// };
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let rate = Rate::hz(2_400_000)?;
///     let ddc_cfg = DdcConfig { frequency_hz: 600e3, decimation: 10, ..DdcConfig::default() };
///     let rate_ddc = Rate::hz(240_000)?;
///     let fm_cfg = FmDemodConfig { audio_decimation: 5, ..FmDemodConfig::default() };
///     let mut mem = vec![0u8; 64 * 1024];
///     let mut arena = ArenaStorage::new(&mut mem);
///     let mut rx = chain!(
///         &mut arena, 256;
///         ConvertBlock::<C32, _>::new(&ConvertConfig::default(), rate, &mut arena)?,
///         DdcBlock::<C32, _>::new(&ddc_cfg, rate, &mut arena)?,
///         FmDemodBlock::<C32, _>::new(&fm_cfg, rate_ddc, &mut arena)?
///     );
///     // Stage 1 is the inner chain (ddc, fm_demod): "1.0" is the ddc.
///     rx.set_param("1.0.frequency_hz", ParamValue::F64(500e3))?;
///     let bytes = vec![128u8; 4800];
///     let mut audio = vec![0.0f32; 100];
///     let p = rx.process(&bytes, &mut audio)?;
///     assert_eq!(p.consumed, 4800);
///     assert!(p.produced > 0);
///     Ok(())
/// }
/// ```
#[macro_export]
macro_rules! chain {
    ($storage:expr, $chunk:expr; $a:expr, $b:expr $(,)?) => {
        $crate::Chain::new($a, $b, $chunk, $storage)?
    };
    ($storage:expr, $chunk:expr; $a:expr, $b:expr, $c:expr $(,)?) => {
        $crate::Chain::new($a, $crate::chain!($storage, $chunk; $b, $c), $chunk, $storage)?
    };
    ($storage:expr, $chunk:expr; $a:expr, $b:expr, $c:expr, $d:expr $(,)?) => {
        $crate::Chain::new($a, $crate::chain!($storage, $chunk; $b, $c, $d), $chunk, $storage)?
    };
    ($storage:expr, $chunk:expr; $a:expr, $b:expr, $c:expr, $d:expr, $e:expr $(,)?) => {
        $crate::Chain::new($a, $crate::chain!($storage, $chunk; $b, $c, $d, $e), $chunk, $storage)?
    };
    ($storage:expr, $chunk:expr; $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr $(,)?) => {
        $crate::Chain::new(
            $a, $crate::chain!($storage, $chunk; $b, $c, $d, $e, $f), $chunk, $storage,
        )?
    };
    ($storage:expr, $chunk:expr; $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr,
     $g:expr $(,)?) => {
        $crate::Chain::new(
            $a, $crate::chain!($storage, $chunk; $b, $c, $d, $e, $f, $g), $chunk, $storage,
        )?
    };
    ($storage:expr, $chunk:expr; $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr,
     $g:expr, $h:expr $(,)?) => {
        $crate::Chain::new(
            $a, $crate::chain!($storage, $chunk; $b, $c, $d, $e, $f, $g, $h), $chunk, $storage,
        )?
    };
}
