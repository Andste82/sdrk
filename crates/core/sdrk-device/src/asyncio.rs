//! Asynchronous stream traits for transports that cannot block (WebUSB, Web Serial, async
//! USB stacks).
//!
//! The traits use `async fn` in traits (stable since Rust 1.75). The futures deliberately
//! have no `Send` bound: futures that wait on browser APIs are not `Send`, and a bound would
//! exclude exactly the drivers these traits are for. This is why both traits carry
//! `#[allow(async_fn_in_trait)]`: the lint warns that callers cannot require `Send`, which
//! is the intended behaviour here. A library does not ship an executor, so there is no
//! adapter from [`AsyncRxStream`] to [`crate::RxStream`].

use crate::{BlockMeta, Rate, SampleFormat, SdrkError};

/// An asynchronous receive stream.
///
/// ```
/// use sdrk_device::{formats, AsyncRxStream, BlockMeta, Error, Rate, SampleFormat};
/// struct Silent;
/// impl AsyncRxStream for Silent {
///     type Error = Error;
///     fn format(&self) -> SampleFormat { formats::CU8 }
///     fn sample_rate(&self) -> Rate { Rate::hz(48_000).unwrap() }
///     async fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta), Error> {
///         buf.fill(128);
///         Ok((buf.len(), BlockMeta::default()))
///     }
/// }
/// assert_eq!(Silent.format(), formats::CU8);
/// ```
#[allow(async_fn_in_trait)]
pub trait AsyncRxStream {
    /// Error type of the driver.
    type Error: SdrkError;
    /// Raw format of the bytes delivered by `read`.
    fn format(&self) -> SampleFormat;
    /// Current sample rate.
    fn sample_rate(&self) -> Rate;
    /// Fills `buf` with whole samples; returns the bytes written and the block metadata.
    async fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta), Self::Error>;
}

/// An asynchronous transmit stream.
#[allow(async_fn_in_trait)]
pub trait AsyncTxStream {
    /// Error type of the driver.
    type Error: SdrkError;
    /// Raw format of the bytes accepted by `write`.
    fn format(&self) -> SampleFormat;
    /// Current sample rate.
    fn sample_rate(&self) -> Rate;
    /// Writes whole samples; returns the bytes accepted.
    async fn write(&mut self, buf: &[u8], meta: &BlockMeta) -> Result<usize, Self::Error>;
    /// Status events reported by the device since the last call.
    async fn poll_status(&mut self) -> Result<Option<BlockMeta>, Self::Error>;
}
