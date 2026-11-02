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
pub trait StreamElem: Copy + Send + 'static + sealed::Sealed {
    /// The port type of the description (`PortType::C32` ...; `PortType::Raw` for `u8`).
    const PORT_TYPE: PortType;
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
    type Ref<'a>;
    /// Writable view of the output data of one call.
    type Mut<'a>;
    /// Number of ports (1 for a stream or a frame, 2 for [`Two`]).
    const COUNT: usize;
}

impl<T: StreamElem> Ports for T {
    type Ref<'a> = &'a [T];
    type Mut<'a> = &'a mut [T];
    const COUNT: usize = 1;
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
}
