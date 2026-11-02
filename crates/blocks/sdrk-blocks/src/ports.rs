//! Port types: what a [`Block`](crate::Block) reads and writes in one `process` call.
//!
//! A port type is a stream of one sample type ([`StreamElem`]), a pair of streams that advance
//! together ([`Two`]) or frames of `f32` values ([`Frame`]). The types exist only at compile
//! time; a stream block keeps the signature it had before ports were introduced.

use sdrk::{C32, CQ15, Q15};

use crate::block::FrameMeta;
use crate::describe::PortType;

mod sealed {
    /// Seals [`StreamElem`](super::StreamElem): only the five sample types implement it.
    pub trait Sealed {}
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
/// use sdrk_blocks::{Ports, StreamElem};
/// struct Packets;
/// impl Ports for Packets {
///     type Ref<'a> = &'a [u8];
///     type Mut<'a> = &'a mut [u8];
///     const COUNT: usize = 1;
///     fn ref_len(view: &Self::Ref<'_>) -> usize { view.len() }
///     fn mut_len(view: &Self::Mut<'_>) -> usize { view.len() }
///     fn ref_skip(view: Self::Ref<'_>, _: usize) -> Self::Ref<'_> { view }
///     fn mut_skip(view: Self::Mut<'_>, _: usize) -> Self::Mut<'_> { view }
///     fn mut_reborrow<'a>(view: &'a mut Self::Mut<'_>) -> Self::Mut<'a> { &mut **view }
/// }
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
}

stream_elem!(C32 => C32, CQ15 => CQ15, f32 => F32, Q15 => Q15, u8 => Raw);

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
    /// Number of ports (1 for a stream or a frame, 2 for [`Two`]).
    const COUNT: usize;

    /// Number of steps a read-only view holds (for [`Two`] the shorter of the two streams).
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
