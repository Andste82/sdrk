//! Block metadata and the synchronous RX/TX stream traits.

use crate::{Rate, SampleFormat, SampleIndex, SdrkError, TimeAnchor};

/// Bit flags describing a block of samples.
///
/// ```
/// use sdrk_device::BlockFlags;
/// let f = BlockFlags::GAP_BEFORE.union(BlockFlags::OVERFLOW);
/// assert!(f.contains(BlockFlags::OVERFLOW));
/// assert!(!f.contains(BlockFlags::LATE));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BlockFlags(u16);

impl BlockFlags {
    /// Samples are missing before this block.
    pub const GAP_BEFORE: Self = Self(1 << 0);
    /// RX: hardware dropped samples.
    pub const OVERFLOW: Self = Self(1 << 1);
    /// TX: hardware ran out of samples.
    pub const UNDERRUN: Self = Self(1 << 2);
    /// First sample of a burst.
    pub const BURST_START: Self = Self(1 << 3);
    /// Last sample of a burst.
    pub const BURST_END: Self = Self(1 << 4);
    /// TX: a timed burst arrived too late.
    pub const LATE: Self = Self(1 << 5);
    /// Frequency, rate or gain changed at the first sample of this block.
    pub const RETUNED: Self = Self(1 << 6);

    const ALL: u16 = 0x7F;

    /// No flag set.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// True if every flag of `other` is set (also true for empty `other`).
    pub const fn contains(&self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of both sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Sets all flags of `other`.
    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    /// Clears all flags of `other`.
    pub fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }

    /// The raw bits.
    pub const fn bits(&self) -> u16 {
        self.0
    }

    /// Builds flags from raw bits, dropping unknown bits.
    pub const fn from_bits_truncate(bits: u16) -> Self {
        Self(bits & Self::ALL)
    }
}

/// Metadata of one block of samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockMeta {
    /// Index of the first sample in this block.
    pub start: SampleIndex,
    /// Flags of the block.
    pub flags: BlockFlags,
    /// Optional absolute time for some sample.
    pub anchor: Option<TimeAnchor>,
    /// TX: requested start time of a burst (unix nanoseconds).
    pub tx_time_ns: Option<i64>,
}

/// A synchronous receive stream. Drivers implement this trait.
pub trait RxStream {
    /// Error type of the driver.
    type Error: SdrkError;
    /// Raw format of the bytes delivered by `read`.
    fn format(&self) -> SampleFormat;
    /// Current sample rate.
    fn sample_rate(&self) -> Rate;
    /// Fills `buf` with whole samples; returns the bytes written and the
    /// metadata of the block. Writes a whole number of samples (and whole
    /// packing groups).
    fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta), Self::Error>;
}

/// A synchronous transmit stream. Drivers implement this trait.
pub trait TxStream {
    /// Error type of the driver.
    type Error: SdrkError;
    /// Raw format of the bytes accepted by `write`.
    fn format(&self) -> SampleFormat;
    /// Current sample rate.
    fn sample_rate(&self) -> Rate;
    /// Writes whole samples; returns the bytes accepted.
    fn write(&mut self, buf: &[u8], meta: &BlockMeta) -> Result<usize, Self::Error>;
    /// Status events reported by the device since the last call (underrun,
    /// late, burst done).
    fn poll_status(&mut self) -> Result<Option<BlockMeta>, Self::Error>;
}
