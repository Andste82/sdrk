//! Raw-format conversion: device bytes into the internal sample types.
//!
//! The effective integer value `e` of every component follows `docs/device.md`
//! (byte order, alignment, sign extension, packing, I/Q swap, planar layout).
//! The mapping to amplitude and to Q15 is exact and identical in the reference
//! and the fallback (see `docs/kernels.md`).

use bytemuck::try_cast_slice;
use sdrk_device::{Container, Endian, FullScale, Layout, SampleFormat};

use crate::error::{Error, Result};
use crate::types::{C32, CQ15, ComplexSample, RealSample};

/// Decodes complex samples from raw bytes (layout `Interleaved` or `Planar`).
///
/// ```
/// use sdrk_core::{kernels::convert::Convert, Fallback, CQ15, Q15};
/// use sdrk_device::formats;
///
/// // One CU8 sample: I = 255, Q = 0 (zero is at 127.5).
/// let mut out = [CQ15::default()];
/// let n = Fallback.convert(&formats::CU8, &[255, 0], &mut out)?;
/// assert_eq!(n, 1);
/// assert_eq!(out[0], CQ15 { re: Q15(32640), im: Q15(-32640) });
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Convert<T: ComplexSample> {
    /// Decode whole samples from `input` (layout Interleaved or Planar) into `output`.
    /// Returns the number of samples written.
    ///
    /// Errors: `InvalidParameter("partial sample")` if `input` does not hold whole
    /// samples and packing groups, `BufferTooSmall` if `output` cannot hold
    /// `format.samples_in_bytes(input.len())` samples, `Unsupported` for `Bits1`,
    /// `InvalidParameter` for a real layout.
    fn convert(&self, format: &SampleFormat, input: &[u8], output: &mut [T]) -> Result<usize>;
}

/// Decodes real samples from raw bytes (layout `Real`).
///
/// ```
/// use sdrk_core::{kernels::convert::ConvertReal, Fallback, Q15};
/// use sdrk_device::formats;
///
/// let mut out = [Q15(0); 2];
/// let n = Fallback.convert_real(&formats::RS16, &[0x00, 0x40, 0x00, 0xC0], &mut out)?;
/// assert_eq!((n, out), (2, [Q15(16384), Q15(-16384)]));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait ConvertReal<T: RealSample> {
    /// Same as [`Convert::convert`] for `Layout::Real`.
    fn convert_real(&self, format: &SampleFormat, input: &[u8], output: &mut [T]) -> Result<usize>;
}

/// Zero-copy view of `bytes` as `CQ15` samples.
///
/// Returns `Some` only if `format` is CS16-compatible (16-bit `I16` container,
/// interleaved, little-endian, no I/Q swap, power-of-two full scale), `bytes`
/// holds whole samples and is aligned for `CQ15` (empty input gives an empty view).
///
/// ```
/// use sdrk_core::{kernels::convert::view_cq15, CQ15, Q15};
/// use sdrk_device::formats;
///
/// let samples = [CQ15 { re: Q15(1), im: Q15(-2) }];
/// let bytes: &[u8] = bytemuck::cast_slice(&samples);
/// assert_eq!(view_cq15(&formats::CS16, bytes), Some(&samples[..]));
/// assert_eq!(view_cq15(&formats::CS8, bytes), None);
/// ```
pub fn view_cq15<'a>(format: &SampleFormat, bytes: &'a [u8]) -> Option<&'a [CQ15]> {
    let compatible = format.container() == Container::I16
        && format.bits() == 16
        && format.layout() == Layout::Interleaved
        && format.endian() == Endian::Little
        && !format.swap_iq()
        && format.full_scale() == FullScale::PowerOfTwo;
    if !compatible {
        return None;
    }
    if bytes.is_empty() {
        return Some(&[]);
    }
    try_cast_slice(bytes).ok()
}

/// Zero-copy view of `bytes` as `C32` samples.
///
/// Returns `Some` only for interleaved little-endian `F32` containers without
/// I/Q swap, if `bytes` holds whole samples and is aligned for `C32` (empty input gives an empty view).
///
/// ```
/// use sdrk_core::{kernels::convert::view_c32, C32};
/// use sdrk_device::formats;
///
/// let samples = [C32::new(0.5, -1.0)];
/// let bytes: &[u8] = bytemuck::cast_slice(&samples);
/// assert_eq!(view_c32(&formats::CF32, bytes), Some(&samples[..]));
/// assert_eq!(view_c32(&formats::CS16, bytes), None);
/// ```
pub fn view_c32<'a>(format: &SampleFormat, bytes: &'a [u8]) -> Option<&'a [C32]> {
    let compatible = format.container() == Container::F32
        && format.layout() == Layout::Interleaved
        && format.endian() == Endian::Little
        && !format.swap_iq();
    if !compatible {
        return None;
    }
    if bytes.is_empty() {
        return Some(&[]);
    }
    try_cast_slice(bytes).ok()
}

/// Shared argument checks of the conversion kernels; returns the sample count.
///
/// Order: wrong layout, `Bits1`, partial samples, output size.
pub(crate) fn plan(
    format: &SampleFormat,
    input_len: usize,
    output_len: usize,
    real: bool,
) -> Result<usize> {
    if (format.layout() == Layout::Real) != real {
        return Err(Error::InvalidParameter(if real {
            "convert_real needs a real layout"
        } else {
            "convert needs an interleaved or planar layout"
        }));
    }
    if format.container() == Container::Bits1 {
        return Err(Error::Unsupported("1-bit containers"));
    }
    let samples = format.samples_in_bytes(input_len);
    if format.bytes_for_samples(samples).ok() != Some(input_len) {
        return Err(Error::InvalidParameter("partial sample"));
    }
    if output_len < samples {
        return Err(Error::BufferTooSmall {
            needed: samples,
            got: output_len,
        });
    }
    Ok(samples)
}

/// Encodes complex samples into raw bytes of a device format (the inverse of [`Convert`]).
///
/// ```
/// use sdrk_core::{kernels::convert::ConvertOut, Fallback, C32};
/// use sdrk_device::formats;
///
/// // Zero is at 127.5 for `U8`: 0.0 rounds to the even neighbour 128, -1.0 gives 0.
/// let mut bytes = [0u8; 2];
/// let n = Fallback.convert_out(&formats::CU8, &[C32::new(0.0, -1.0)], &mut bytes)?;
/// assert_eq!((n, bytes), (2, [128, 0]));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait ConvertOut<T: ComplexSample> {
    /// Encode `input` into the raw bytes of `format` (layout Interleaved or Planar) and return
    /// the number of bytes written.
    ///
    /// From `C32`: `v = x * full_scale + offset` in f64 (NaN counts as amplitude 0), the raw
    /// value is `v` rounded to the nearest integer, ties to even, saturated to the range of the
    /// format; `F32` containers copy the bits. From `CQ15` the integer rules of
    /// `docs/kernels.md` apply. Alignment, packing, I/Q swap, planar halves and byte order are
    /// the inverse of [`Convert`].
    ///
    /// Errors: `BufferTooSmall` if `output` cannot hold the bytes, `Unsupported` for `Bits1`,
    /// `InvalidParameter` for a real layout or a sample count that does not fill whole packing
    /// groups.
    fn convert_out(&self, format: &SampleFormat, input: &[T], output: &mut [u8]) -> Result<usize>;
}

/// Encodes real samples into raw bytes (layout `Real`).
///
/// ```
/// use sdrk_core::{kernels::convert::ConvertOutReal, Fallback, Q15};
/// use sdrk_device::formats;
///
/// let mut bytes = [0u8; 4];
/// let n = Fallback.convert_out_real(&formats::RS16, &[Q15(16384), Q15(-16384)], &mut bytes)?;
/// assert_eq!((n, bytes), (4, [0x00, 0x40, 0x00, 0xC0]));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait ConvertOutReal<T: RealSample> {
    /// Same as [`ConvertOut::convert_out`] for `Layout::Real`.
    fn convert_out_real(
        &self,
        format: &SampleFormat,
        input: &[T],
        output: &mut [u8],
    ) -> Result<usize>;
}

/// Shared argument checks of the encoding kernels; returns the number of bytes to write.
///
/// Order: wrong layout, `Bits1`, incomplete packing group, output size.
pub(crate) fn plan_out(
    format: &SampleFormat,
    samples: usize,
    output_len: usize,
    real: bool,
) -> Result<usize> {
    if (format.layout() == Layout::Real) != real {
        return Err(Error::InvalidParameter(if real {
            "convert_out_real needs a real layout"
        } else {
            "convert_out needs an interleaved or planar layout"
        }));
    }
    if format.container() == Container::Bits1 {
        return Err(Error::Unsupported("1-bit containers"));
    }
    let bytes = format
        .bytes_for_samples(samples)
        .map_err(|_| Error::InvalidParameter("sample count does not fill whole packing groups"))?;
    if output_len < bytes {
        return Err(Error::BufferTooSmall {
            needed: bytes,
            got: output_len,
        });
    }
    Ok(bytes)
}
