//! Port types: what a [`Block`](crate::Block) reads and writes in one `process` call.
//!
//! A port type is a stream of one sample type ([`StreamElem`]), a pair of streams that advance
//! together ([`Two`]), frames of `f32` values ([`Frame`]), packets ([`Packets`]) or a clock
//! stream with packets ([`Clocked`]). The types exist only at compile time; a stream block
//! keeps the signature it had before ports were introduced.

use core::mem::{align_of, size_of};

use sdrk::{C32, CQ15, Q15};
use sdrk_proto::Packet;

use crate::block::FrameMeta;
use crate::describe::PortType;
use crate::error::Result;
use crate::storage::Storage;

pub(crate) mod sealed {
    use super::Ports;
    use crate::error::Result;
    use crate::storage::Storage;

    /// Seals [`StreamElem`](super::StreamElem): only the five sample types implement it.
    pub trait Sealed {}

    /// Seals [`ChainEnd`](super::ChainEnd): stream elements, [`Two`](super::Two) and
    /// [`Frame`](super::Frame) implement it.
    pub trait SealedEnd {}

    /// Seals [`ChainItem`](super::ChainItem) and holds its machinery, which stays private to
    /// the crate: the intermediate buffer(s) of a chain and their views.
    pub trait ChainBufs: Ports {
        /// The buffer(s) that hold `len` items: one buffer for a stream, two for a pair.
        type Bufs<S: Storage>: Send;

        /// Bytes [`ChainBufs::alloc`] takes for `len` items on an empty arena that starts at
        /// an [`ARENA_ALIGN`](crate::ARENA_ALIGN)-aligned address, including the padding
        /// between the buffers of a pair (saturating).
        fn bytes(len: usize) -> usize;

        /// Allocates the buffer(s) of `len` items from `storage`.
        fn alloc<S: Storage>(storage: &mut S, len: usize) -> Result<Self::Bufs<S>>;

        /// Number of items the buffer(s) hold.
        fn capacity<S: Storage>(bufs: &Self::Bufs<S>) -> usize;

        /// Read-only view of the items `start..end`; `None` if the range is out of bounds.
        fn view<S: Storage>(
            bufs: &Self::Bufs<S>,
            start: usize,
            end: usize,
        ) -> Option<Self::Ref<'_>>;

        /// Writable view of the items from `start` to the end; `None` if `start` is out of
        /// bounds.
        fn view_mut_from<S: Storage>(
            bufs: &mut Self::Bufs<S>,
            start: usize,
        ) -> Option<Self::Mut<'_>>;

        /// Moves the items `start..end` to the front (of every buffer, in lockstep); `None` if
        /// the range is out of bounds.
        fn move_to_front<S: Storage>(
            bufs: &mut Self::Bufs<S>,
            start: usize,
            end: usize,
        ) -> Option<()>;
    }
}

/// Moves `s[start..end]` to the front of `s`; `None` if the range is out of bounds.
fn move_front<T: Copy>(s: &mut [T], start: usize, end: usize) -> Option<()> {
    if start > end || end > s.len() {
        return None;
    }
    s.copy_within(start..end, 0);
    Some(())
}

macro_rules! stream_elem {
    ($($ty:ty => $port:ident),* $(,)?) => {$(
        impl sealed::Sealed for $ty {}
        impl StreamElem for $ty {
            const PORT_TYPE: PortType = PortType::$port;
            fn view(samples: &[Self]) -> <Self as Ports>::Ref<'_> {
                samples
            }
            fn view_mut(samples: &mut [Self]) -> <Self as Ports>::Mut<'_> {
                samples
            }
            fn slice(view: <Self as Ports>::Ref<'_>) -> &[Self] {
                view
            }
        }
    )*};
}

/// The element of a stream port. Sealed: implemented for `C32`, `CQ15`, `f32`, `Q15` and `u8`
/// only.
///
/// ```
/// use sdrk::C32;
/// use sdrk_blocks::{PortType, StreamElem};
/// assert_eq!(<C32 as StreamElem>::PORT_TYPE, PortType::C32);
/// assert_eq!(<u8 as StreamElem>::PORT_TYPE, PortType::Raw);
/// ```
///
/// The elements are plain data (`bytemuck::Pod`) with a `Default`, so a chain can allocate
/// intermediate buffers of them. A port type that is not a stream element (a packet port) is
/// rejected where a stream is required:
///
/// ```compile_fail,E0277
/// use sdrk_blocks::{Packets, StreamElem};
/// fn needs_stream<T: StreamElem>() {}
/// needs_stream::<Packets>();
/// ```
pub trait StreamElem: Copy + Send + Default + bytemuck::Pod + 'static + sealed::Sealed {
    /// The port type of the description (`PortType::C32` ...; `PortType::Raw` for `u8`).
    const PORT_TYPE: PortType;

    /// A slice as the read-only port view of this element (the identity; lets generic code
    /// such as `Chain` name the view of a stream).
    fn view(samples: &[Self]) -> <Self as Ports>::Ref<'_>;

    /// A mutable slice as the writable port view of this element (the identity).
    fn view_mut(samples: &mut [Self]) -> <Self as Ports>::Mut<'_>;

    /// The read-only port view of this element as a slice (the identity; the inverse of
    /// [`StreamElem::view`]).
    ///
    /// ```
    /// use sdrk_blocks::StreamElem;
    /// let data = [1.0f32, 2.0];
    /// assert_eq!(<f32 as StreamElem>::slice(<f32 as StreamElem>::view(&data)), &data);
    /// ```
    fn slice(view: <Self as Ports>::Ref<'_>) -> &[Self];
}

stream_elem!(C32 => C32, CQ15 => CQ15, f32 => F32, Q15 => Q15, u8 => Raw);

/// A port type that a [`Chain`](crate::Chain) can pass between its stages: a stream element
/// ([`StreamElem`]) or a pair of streams ([`Two`]) of stream elements. Sealed.
///
/// The chain keeps the intermediate items in buffers from a [`Storage`]: one buffer of
/// `chunk` samples for a stream, two buffers of `chunk` samples for a pair, written and read
/// in lockstep (the rule of [`Two`]).
///
/// ```
/// use sdrk::C32;
/// use sdrk_blocks::{ChainItem, Two};
/// fn passable<T: ChainItem>() {}
/// passable::<C32>();
/// passable::<Two<f32, C32>>();
/// ```
///
/// Frames and packets (which are not stream elements) cannot be passed between the stages of
/// a chain:
///
/// ```compile_fail,E0277
/// use sdrk_blocks::{ChainItem, Frame};
/// fn passable<T: ChainItem>() {}
/// passable::<Frame>();
/// ```
///
/// ```compile_fail,E0277
/// use sdrk_blocks::{ChainItem, Packets};
/// fn passable<T: ChainItem>() {}
/// passable::<Packets>();
/// ```
pub trait ChainItem: Ports + sealed::ChainBufs {}

impl<T: StreamElem> sealed::ChainBufs for T {
    type Bufs<S: Storage> = S::Buf<T>;

    fn bytes(len: usize) -> usize {
        len.saturating_mul(size_of::<T>())
    }
    fn alloc<S: Storage>(storage: &mut S, len: usize) -> Result<Self::Bufs<S>> {
        storage.alloc::<T>(len)
    }
    fn capacity<S: Storage>(bufs: &Self::Bufs<S>) -> usize {
        bufs.as_ref().len()
    }
    fn view<S: Storage>(
        bufs: &Self::Bufs<S>,
        start: usize,
        end: usize,
    ) -> Option<<T as Ports>::Ref<'_>> {
        bufs.as_ref().get(start..end).map(T::view)
    }
    fn view_mut_from<S: Storage>(
        bufs: &mut Self::Bufs<S>,
        start: usize,
    ) -> Option<<T as Ports>::Mut<'_>> {
        bufs.as_mut().get_mut(start..).map(T::view_mut)
    }
    fn move_to_front<S: Storage>(bufs: &mut Self::Bufs<S>, start: usize, end: usize) -> Option<()> {
        move_front(bufs.as_mut(), start, end)
    }
}

impl<T: StreamElem> ChainItem for T {}

/// A port type that may be the input or the output of a [`Chain`](crate::Chain): a stream
/// element ([`StreamElem`]), a pair of streams ([`Two`]) or frames ([`Frame`]). Sealed.
///
/// Packet ports ([`Packets`], [`Clocked`]) are not chain ends: a chain never starts with an
/// encoder or ends with a decoder (the pipeline connects those blocks).
///
/// ```
/// use sdrk::C32;
/// use sdrk_blocks::{ChainEnd, Frame, Two};
/// fn end<T: ChainEnd>() {}
/// end::<C32>();
/// end::<Two<f32, f32>>();
/// end::<Frame>();
/// ```
///
/// ```compile_fail,E0277
/// use sdrk_blocks::{ChainEnd, Packets};
/// fn end<T: ChainEnd>() {}
/// end::<Packets>();
/// ```
pub trait ChainEnd: Ports + sealed::SealedEnd {}

impl<T: StreamElem> sealed::SealedEnd for T {}
impl<T: StreamElem> ChainEnd for T {}
impl<X: StreamElem, Y: StreamElem> sealed::SealedEnd for Two<X, Y> {}
impl<X: StreamElem, Y: StreamElem> ChainEnd for Two<X, Y> {}
impl sealed::SealedEnd for Frame {}
impl ChainEnd for Frame {}

impl<X: StreamElem, Y: StreamElem> sealed::ChainBufs for Two<X, Y> {
    type Bufs<S: Storage> = (S::Buf<X>, S::Buf<Y>);

    fn bytes(len: usize) -> usize {
        let first = len.saturating_mul(size_of::<X>());
        let align = align_of::<Y>();
        let pad = (align - first % align) % align;
        first
            .saturating_add(pad)
            .saturating_add(len.saturating_mul(size_of::<Y>()))
    }
    fn alloc<S: Storage>(storage: &mut S, len: usize) -> Result<Self::Bufs<S>> {
        let x = storage.alloc::<X>(len)?;
        let y = storage.alloc::<Y>(len)?;
        Ok((x, y))
    }
    fn capacity<S: Storage>(bufs: &Self::Bufs<S>) -> usize {
        bufs.0.as_ref().len().min(bufs.1.as_ref().len())
    }
    fn view<S: Storage>(bufs: &Self::Bufs<S>, start: usize, end: usize) -> Option<(&[X], &[Y])> {
        Some((
            bufs.0.as_ref().get(start..end)?,
            bufs.1.as_ref().get(start..end)?,
        ))
    }
    fn view_mut_from<S: Storage>(
        bufs: &mut Self::Bufs<S>,
        start: usize,
    ) -> Option<(&mut [X], &mut [Y])> {
        let (x, y) = bufs;
        Some((x.as_mut().get_mut(start..)?, y.as_mut().get_mut(start..)?))
    }
    fn move_to_front<S: Storage>(bufs: &mut Self::Bufs<S>, start: usize, end: usize) -> Option<()> {
        move_front(bufs.0.as_mut(), start, end)?;
        move_front(bufs.1.as_mut(), start, end)
    }
}

impl<X: StreamElem, Y: StreamElem> ChainItem for Two<X, Y> {}

/// What a block reads and writes in one `process` call.
///
/// A stream element `T` is its own port type: the input is `&[T]`, the output `&mut [T]`.
///
/// ```
/// use sdrk::C32;
/// use sdrk_blocks::{Frame, Ports, Two};
/// assert_eq!(<C32 as Ports>::COUNT, 1);
/// assert_eq!(<Two<f32, f32> as Ports>::COUNT, 2);
/// assert_eq!(<Frame as Ports>::COUNT, 1);
/// // The views of a stream are plain slices.
/// let data = [C32::new(1.0, 0.0)];
/// let view: <C32 as Ports>::Ref<'_> = &data;
/// assert_eq!(view.len(), 1);
/// ```
pub trait Ports: 'static {
    /// Read-only view of the input (or output) data of one call.
    type Ref<'a>: Copy;
    /// Writable view of the output data of one call.
    type Mut<'a>;
    /// Number of ports (1 for a stream, a frame or packets, 2 for [`Two`] and [`Clocked`]).
    const COUNT: usize;

    /// Number of steps a read-only view holds (for [`Two`] the shorter of the two streams, for
    /// [`Clocked`] the clock samples).
    fn ref_len(view: &Self::Ref<'_>) -> usize;

    /// Number of steps a writable view can still take (a stream: its length; [`Two`]: the
    /// shorter stream; [`Frame`]: 1 while a frame fits, 0 after [`Ports::mut_skip`]).
    fn mut_len(view: &Self::Mut<'_>) -> usize;

    /// The read-only view without its first `n` steps (all streams of [`Two`] together). `n`
    /// beyond the length gives an empty view.
    fn ref_skip(view: Self::Ref<'_>, n: usize) -> Self::Ref<'_>;

    /// The writable view without its first `n` steps (all streams of [`Two`] together). `n`
    /// beyond the length gives an empty view; a [`Frame`] view with `n >= 1` has no capacity
    /// left (`mut_len` 0, empty values, `meta` kept).
    fn mut_skip(view: Self::Mut<'_>, n: usize) -> Self::Mut<'_>;

    /// A shorter-lived writable view of the same data; the original is usable again after it.
    fn mut_reborrow<'a>(view: &'a mut Self::Mut<'_>) -> Self::Mut<'a>;
}

fn skip_ref<T>(s: &[T], n: usize) -> &[T] {
    s.get(n..).unwrap_or_default()
}

fn skip_mut<T>(s: &mut [T], n: usize) -> &mut [T] {
    let n = n.min(s.len());
    s.split_at_mut_checked(n)
        .map(|(_, rest)| rest)
        .unwrap_or_default()
}

impl<T: StreamElem> Ports for T {
    type Ref<'a> = &'a [T];
    type Mut<'a> = &'a mut [T];
    const COUNT: usize = 1;

    fn ref_len(view: &Self::Ref<'_>) -> usize {
        view.len()
    }
    fn mut_len(view: &Self::Mut<'_>) -> usize {
        view.len()
    }
    fn ref_skip(view: Self::Ref<'_>, n: usize) -> Self::Ref<'_> {
        skip_ref(view, n)
    }
    fn mut_skip(view: Self::Mut<'_>, n: usize) -> Self::Mut<'_> {
        skip_mut(view, n)
    }
    fn mut_reborrow<'a>(view: &'a mut Self::Mut<'_>) -> Self::Mut<'a> {
        &mut **view
    }
}

/// Two streams that advance together (one sample of each per step).
///
/// Lockstep rule: both input slices of a call have the same length and the block consumes the
/// same number of samples from each (`Progress::consumed` counts pairs). Both output slices
/// have the same length and the block writes the same number of samples to each
/// (`Progress::produced` counts pairs). Unequal input or output lengths are `LengthMismatch`
/// and change nothing.
///
/// ```
/// use sdrk_blocks::{Ports, Two};
/// let re = [1.0f32, 2.0];
/// let im = [3.0f32, 4.0];
/// let input: <Two<f32, f32> as Ports>::Ref<'_> = (&re, &im);
/// assert_eq!(input.0.len(), input.1.len());
/// assert_eq!(<Two<f32, f32> as Ports>::COUNT, 2);
/// ```
pub struct Two<A: StreamElem, B: StreamElem>(core::marker::PhantomData<(A, B)>);

impl<A: StreamElem, B: StreamElem> Ports for Two<A, B> {
    type Ref<'a> = (&'a [A], &'a [B]);
    type Mut<'a> = (&'a mut [A], &'a mut [B]);
    const COUNT: usize = 2;

    fn ref_len(view: &Self::Ref<'_>) -> usize {
        view.0.len().min(view.1.len())
    }
    fn mut_len(view: &Self::Mut<'_>) -> usize {
        view.0.len().min(view.1.len())
    }
    fn ref_skip(view: Self::Ref<'_>, n: usize) -> Self::Ref<'_> {
        (skip_ref(view.0, n), skip_ref(view.1, n))
    }
    fn mut_skip(view: Self::Mut<'_>, n: usize) -> Self::Mut<'_> {
        (skip_mut(view.0, n), skip_mut(view.1, n))
    }
    fn mut_reborrow<'a>(view: &'a mut Self::Mut<'_>) -> Self::Mut<'a> {
        (&mut *view.0, &mut *view.1)
    }
}

/// Frames of `f32` values (spectra, measurements). Output only.
///
/// The writable view is a [`FrameOut`]; the read-only view is a slice of `f32` and is not used
/// by any block yet.
///
/// ```
/// use sdrk_blocks::{Frame, Ports};
/// assert_eq!(<Frame as Ports>::COUNT, 1);
/// ```
pub struct Frame;

/// The output of a [`Frame`] port in one call.
///
/// ```
/// use sdrk_blocks::{FrameMeta, FrameOut};
/// let mut values = [0.0f32; 4];
/// let mut meta = None;
/// let out = FrameOut { values: &mut values, meta: &mut meta };
/// out.values.fill(1.0);
/// *out.meta = Some(FrameMeta { first_sample: 0, averaged: 1, exponent: 0 });
/// assert_eq!(values, [1.0; 4]);
/// assert!(meta.is_some());
/// ```
pub struct FrameOut<'a> {
    /// Exactly `Block::frame_len()` values; written when a frame completes.
    pub values: &'a mut [f32],
    /// Set to `Some` when the call completed a frame, left unchanged otherwise (the caller
    /// sets it to `None` before the call).
    pub meta: &'a mut Option<FrameMeta>,
}

impl Ports for Frame {
    type Ref<'a> = &'a [f32];
    type Mut<'a> = FrameOut<'a>;
    const COUNT: usize = 1;

    fn ref_len(view: &Self::Ref<'_>) -> usize {
        view.len()
    }
    fn mut_len(view: &Self::Mut<'_>) -> usize {
        usize::from(!view.values.is_empty())
    }
    fn ref_skip(view: Self::Ref<'_>, n: usize) -> Self::Ref<'_> {
        skip_ref(view, n)
    }
    fn mut_skip(view: Self::Mut<'_>, n: usize) -> Self::Mut<'_> {
        if n == 0 {
            return view;
        }
        FrameOut {
            values: Default::default(),
            meta: view.meta,
        }
    }
    fn mut_reborrow<'a>(view: &'a mut Self::Mut<'_>) -> Self::Mut<'a> {
        FrameOut {
            values: &mut *view.values,
            meta: &mut *view.meta,
        }
    }
}

/// Packets ([`Packet`]: bytes with a sample index and flags), one item per packet.
///
/// The views are slices of packets: `Ref = &[Packet]`, `Mut = &mut [Packet]`. `Progress`
/// counts packets on a packet port. `Packet` is not a [`StreamElem`], so [`Chain`](crate::Chain)
/// and [`Fork`](crate::Fork) do not accept packet ports (a compile error); the pipeline
/// connects packet blocks.
///
/// ```
/// use sdrk_blocks::{Packet, Packets, Ports};
/// assert_eq!(<Packets as Ports>::COUNT, 1);
/// let queue = [Packet::default(); 3];
/// let view: <Packets as Ports>::Ref<'_> = &queue;
/// assert_eq!(<Packets as Ports>::ref_len(&<Packets as Ports>::ref_skip(view, 1)), 2);
/// ```
pub struct Packets;

impl Ports for Packets {
    type Ref<'a> = &'a [Packet];
    type Mut<'a> = &'a mut [Packet];
    const COUNT: usize = 1;

    fn ref_len(view: &Self::Ref<'_>) -> usize {
        view.len()
    }
    fn mut_len(view: &Self::Mut<'_>) -> usize {
        view.len()
    }
    fn ref_skip(view: Self::Ref<'_>, n: usize) -> Self::Ref<'_> {
        skip_ref(view, n)
    }
    fn mut_skip(view: Self::Mut<'_>, n: usize) -> Self::Mut<'_> {
        skip_mut(view, n)
    }
    fn mut_reborrow<'a>(view: &'a mut Self::Mut<'_>) -> Self::Mut<'a> {
        &mut **view
    }
}

/// A clock stream `A` (port a) with packets (port b): the input of an encoder that is driven
/// by a sample clock. Implemented for `Clocked<A, Packets>`.
///
/// Lockstep applies to port a only: `process` gets `n` clock samples and the queued packets
/// (any number, also none). `Progress::consumed` counts clock samples;
/// [`Block::consumed_packets`](crate::Block::consumed_packets) reports how many packets
/// (from the front of the queue) the same call consumed. The caller resubmits the
/// unconsumed clock samples and packets. The values of the clock samples may be ignored.
/// Port b carries no clock domain and no group delay.
///
/// [`Ports::ref_len`] and [`Ports::ref_skip`] work on the clock samples; the packets stay
/// untouched (they advance by `consumed_packets`).
///
/// ```
/// use sdrk_blocks::{Clocked, Packet, Packets, Ports};
/// assert_eq!(<Clocked<f32, Packets> as Ports>::COUNT, 2);
/// let clock = [0.0f32; 4];
/// let queue = [Packet::default(); 1];
/// let view: <Clocked<f32, Packets> as Ports>::Ref<'_> = (&clock, &queue);
/// assert_eq!(<Clocked<f32, Packets> as Ports>::ref_len(&view), 4);
/// let rest = <Clocked<f32, Packets> as Ports>::ref_skip(view, 3);
/// assert_eq!((rest.0.len(), rest.1.len()), (1, 1));
/// ```
pub struct Clocked<A: StreamElem, B>(core::marker::PhantomData<(A, B)>);

impl<A: StreamElem> Ports for Clocked<A, Packets> {
    type Ref<'a> = (&'a [A], &'a [Packet]);
    type Mut<'a> = (&'a mut [A], &'a mut [Packet]);
    const COUNT: usize = 2;

    fn ref_len(view: &Self::Ref<'_>) -> usize {
        view.0.len()
    }
    fn mut_len(view: &Self::Mut<'_>) -> usize {
        view.0.len()
    }
    fn ref_skip(view: Self::Ref<'_>, n: usize) -> Self::Ref<'_> {
        (skip_ref(view.0, n), view.1)
    }
    fn mut_skip(view: Self::Mut<'_>, n: usize) -> Self::Mut<'_> {
        (skip_mut(view.0, n), view.1)
    }
    fn mut_reborrow<'a>(view: &'a mut Self::Mut<'_>) -> Self::Mut<'a> {
        (&mut *view.0, &mut *view.1)
    }
}
