//! Raw sample formats and their byte-level properties.

use crate::{Error, Result};

/// Storage container of one component (I, Q or a real value).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Container {
    /// Unsigned 8-bit.
    U8,
    /// Signed 8-bit.
    I8,
    /// Signed 16-bit.
    I16,
    /// Signed 32-bit.
    I32,
    /// 32-bit float.
    F32,
    /// Two 12-bit values in 3 bytes.
    Packed12,
    /// Four 10-bit values in 5 bytes.
    Packed10,
    /// Eight 1-bit values per byte.
    Bits1,
}

/// Arrangement of the components of a sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// I and Q alternate: I0, Q0, I1, Q1, ...
    Interleaved,
    /// All I values, then all Q values.
    Planar,
    /// One real value per sample.
    Real,
}

/// Position of the valid bits inside the container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Align {
    /// Valid bits are the low bits.
    Lsb,
    /// Valid bits are the high bits.
    Msb,
}

/// Byte order of multi-byte containers in the input data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endian {
    /// Little endian.
    Little,
    /// Big endian.
    Big,
}

/// What raw value corresponds to amplitude 1.0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FullScale {
    /// `2^(bits-1)`, the normal case.
    PowerOfTwo,
    /// An explicit integer, e.g. 2048 for bladeRF `SC16_Q11`.
    Explicit(u32),
}

/// Description of the raw bytes a device delivers or accepts.
///
/// Semantics (implemented by the conversion kernels of a later milestone):
///
/// - **Value mapping:** amplitude = (raw - offset) / `full_scale`. The offset
///   is 127.5 for [`Container::U8`] and 0 otherwise. For [`Align::Msb`] the
///   effective value is `raw >> (container_bits - bits)` (arithmetic shift);
///   for [`Align::Lsb`] it is the low `bits` bits, sign-extended.
/// - **`swap_iq`:** if true, the first value of each pair is Q.
/// - **Planar:** the buffer holds all I values followed by all Q values; the
///   length is even and both halves are equal.
/// - **`Endian::Big`:** affects only the byte order of multi-byte containers in
///   the input data; the host is always little-endian.
///
/// ```
/// use sdrk_device::formats;
/// assert_eq!(formats::CS16.bytes_for_samples(1000)?, 4000);
/// # Ok::<(), sdrk_device::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SampleFormat {
    container: Container,
    bits: u8,
    signed: bool,
    layout: Layout,
    align: Align,
    endian: Endian,
    swap_iq: bool,
    full_scale: FullScale,
}

impl SampleFormat {
    /// Creates a validated format with default `swap_iq = false` and
    /// [`FullScale::PowerOfTwo`].
    pub fn new(
        container: Container,
        bits: u8,
        signed: bool,
        layout: Layout,
        align: Align,
        endian: Endian,
    ) -> Result<Self> {
        let (max_bits, exact) = match container {
            Container::U8 | Container::I8 => (8, false),
            Container::I16 => (16, false),
            Container::I32 => (32, false),
            Container::F32 => (32, true),
            Container::Packed12 => (12, true),
            Container::Packed10 => (10, true),
            Container::Bits1 => (1, true),
        };
        if bits == 0 {
            return Err(Error::InvalidFormat("bits must be at least 1"));
        }
        if bits > max_bits || (exact && bits != max_bits) {
            return Err(Error::InvalidFormat("bits do not fit the container"));
        }
        match container {
            Container::U8 if signed => {
                return Err(Error::InvalidFormat("u8 containers must be unsigned"));
            }
            Container::U8 => {}
            _ if !signed => {
                return Err(Error::InvalidFormat("only u8 containers may be unsigned"));
            }
            _ => {}
        }
        if layout == Layout::Planar
            && matches!(
                container,
                Container::Packed12 | Container::Packed10 | Container::Bits1
            )
        {
            return Err(Error::InvalidFormat("packed containers cannot be planar"));
        }
        Ok(Self::new_const(
            container, bits, signed, layout, align, endian,
        ))
    }

    /// Unchecked constructor for the constants in [`formats`].
    const fn new_const(
        container: Container,
        bits: u8,
        signed: bool,
        layout: Layout,
        align: Align,
        endian: Endian,
    ) -> Self {
        Self {
            container,
            bits,
            signed,
            layout,
            align,
            endian,
            swap_iq: false,
            full_scale: FullScale::PowerOfTwo,
        }
    }

    const fn with_swap_iq_const(mut self, swap: bool) -> Self {
        self.swap_iq = swap;
        self
    }

    const fn with_full_scale_const(mut self, fs: FullScale) -> Self {
        self.full_scale = fs;
        self
    }

    /// Returns a copy with I/Q order swapped (first value of each pair is Q).
    #[must_use]
    pub const fn with_swap_iq(self, swap: bool) -> Self {
        self.with_swap_iq_const(swap)
    }

    /// Returns a copy with another full-scale definition; `Explicit(0)` is rejected.
    pub fn with_full_scale(self, fs: FullScale) -> Result<Self> {
        if fs == FullScale::Explicit(0) {
            return Err(Error::InvalidFormat("explicit full scale must be positive"));
        }
        Ok(self.with_full_scale_const(fs))
    }

    /// Storage container.
    pub const fn container(&self) -> Container {
        self.container
    }
    /// Effective bits per component.
    pub const fn bits(&self) -> u8 {
        self.bits
    }
    /// Whether values are signed.
    pub const fn signed(&self) -> bool {
        self.signed
    }
    /// Component layout.
    pub const fn layout(&self) -> Layout {
        self.layout
    }
    /// Bit alignment inside the container.
    pub const fn align(&self) -> Align {
        self.align
    }
    /// Byte order of the data.
    pub const fn endian(&self) -> Endian {
        self.endian
    }
    /// Whether I and Q are swapped.
    pub const fn swap_iq(&self) -> bool {
        self.swap_iq
    }
    /// Full-scale definition.
    pub const fn full_scale(&self) -> FullScale {
        self.full_scale
    }

    /// Number of values per sample: 2 for interleaved/planar, 1 for real.
    pub const fn values_per_sample(&self) -> usize {
        match self.layout {
            Layout::Interleaved | Layout::Planar => 2,
            Layout::Real => 1,
        }
    }

    /// Exact byte count for `n` samples.
    ///
    /// Packed containers are LSB-first and little-endian, values in the order
    /// I0, Q0, I1, Q1, ... `Packed12` packs 2 values into 3 bytes, `Packed10`
    /// 4 values into 5 bytes, `Bits1` 8 values per byte. Returns
    /// `InvalidFormat` if `n` does not fill whole groups and `Overflow` if the
    /// result does not fit `usize`.
    ///
    /// ```
    /// use sdrk_device::formats;
    /// assert_eq!(formats::PACKED12.bytes_for_samples(1000)?, 3000);
    /// assert!(formats::PACKED10.bytes_for_samples(999).is_err());
    /// # Ok::<(), sdrk_device::Error>(())
    /// ```
    pub fn bytes_for_samples(&self, n: usize) -> Result<usize> {
        let values = n
            .checked_mul(self.values_per_sample())
            .ok_or(Error::Overflow("sample count"))?;
        let (group_values, group_bytes) = self.group();
        if values % group_values != 0 {
            return Err(Error::InvalidFormat(
                "sample count does not fill whole packing groups",
            ));
        }
        (values / group_values)
            .checked_mul(group_bytes)
            .ok_or(Error::Overflow("byte count"))
    }

    /// Number of whole samples contained in `bytes`.
    ///
    /// ```
    /// use sdrk_device::formats;
    /// assert_eq!(formats::CU8.samples_in_bytes(2001), 1000);
    /// ```
    pub fn samples_in_bytes(&self, bytes: usize) -> usize {
        let (group_values, group_bytes) = self.group();
        let values = (bytes / group_bytes).saturating_mul(group_values);
        values / self.values_per_sample()
    }

    /// Values and bytes of one packing group (one value for unpacked containers).
    const fn group(&self) -> (usize, usize) {
        match self.container {
            Container::U8 | Container::I8 => (1, 1),
            Container::I16 => (1, 2),
            Container::I32 | Container::F32 => (1, 4),
            Container::Packed12 => (2, 3),
            Container::Packed10 => (4, 5),
            Container::Bits1 => (8, 1),
        }
    }

    /// The raw value that corresponds to amplitude 1.0.
    ///
    /// `2^(bits-1)` for [`FullScale::PowerOfTwo`] (128 for `U8`), the explicit
    /// value otherwise, and 1.0 for `F32`.
    pub fn full_scale_value(&self) -> f64 {
        if self.container == Container::F32 {
            return 1.0;
        }
        match self.full_scale {
            FullScale::Explicit(v) => f64::from(v),
            FullScale::PowerOfTwo => {
                (1u64 << u32::from(self.bits.saturating_sub(1)).min(63)) as f64
            }
        }
    }
}

/// Well-known formats and the devices they come from.
pub mod formats {
    use super::{Align, Container, Endian, FullScale, Layout, SampleFormat};

    const fn mk(
        c: Container,
        bits: u8,
        signed: bool,
        layout: Layout,
        align: Align,
    ) -> SampleFormat {
        SampleFormat::new_const(c, bits, signed, layout, align, Endian::Little)
    }

    /// RTL-SDR: unsigned 8-bit interleaved, zero at 127.5.
    pub const CU8: SampleFormat = mk(Container::U8, 8, false, Layout::Interleaved, Align::Lsb);
    /// HackRF: signed 8-bit interleaved.
    pub const CS8: SampleFormat = mk(Container::I8, 8, true, Layout::Interleaved, Align::Lsb);
    /// Airspy HF+, LR2021 IQ16: signed 16-bit interleaved (I in the low half-word).
    pub const CS16: SampleFormat = mk(Container::I16, 16, true, Layout::Interleaved, Align::Lsb);
    /// PlutoSDR/AD9361: 12-bit values sign-extended in 16 bits.
    pub const CS12_LSB16: SampleFormat =
        mk(Container::I16, 12, true, Layout::Interleaved, Align::Lsb);
    /// bladeRF `SC16_Q11`: 12-bit values in 16 bits, full scale 2048.
    pub const SC16_Q11: SampleFormat =
        mk(Container::I16, 12, true, Layout::Interleaved, Align::Lsb)
            .with_full_scale_const(FullScale::Explicit(2048));
    /// `SDRplay` API style: all I values, then all Q values, signed 16-bit.
    pub const CS16_PLANAR: SampleFormat = mk(Container::I16, 16, true, Layout::Planar, Align::Lsb);
    /// LimeSDR-style wire packing: two 12-bit values in 3 bytes.
    pub const PACKED12: SampleFormat = mk(
        Container::Packed12,
        12,
        true,
        Layout::Interleaved,
        Align::Lsb,
    );
    /// ESP-SDR CAP20-style packing: four 10-bit values in 5 bytes (exact
    /// device bit order to be confirmed by its driver).
    pub const PACKED10: SampleFormat = mk(
        Container::Packed10,
        10,
        true,
        Layout::Interleaved,
        Align::Lsb,
    );
    /// SX1255 mode B over I2S: 24 valid bits, MSB-aligned in 32 (example).
    pub const I2S32_MSB: SampleFormat =
        mk(Container::I32, 24, true, Layout::Interleaved, Align::Msb);
    /// Files, SoapySDR `CF32`: interleaved 32-bit float.
    pub const CF32: SampleFormat = mk(Container::F32, 32, true, Layout::Interleaved, Align::Lsb);
    /// Real-valued 32-bit float (audio, real sampling).
    pub const RF32: SampleFormat = mk(Container::F32, 32, true, Layout::Real, Align::Lsb);
    /// PCM audio: real signed 16-bit.
    pub const RS16: SampleFormat = mk(Container::I16, 16, true, Layout::Real, Align::Lsb);
}
