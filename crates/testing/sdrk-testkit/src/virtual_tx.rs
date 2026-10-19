//! A virtual transmitter: records what is written and injects device statuses.

use sdrk_core::Error;
use sdrk_device::{BlockFlags, BlockMeta, Rate, SampleFormat, SampleIndex, TxStream};

/// Maximum number of recorded writes.
pub const MAX_RECORDS: usize = 64;

/// One recorded call of [`TxStream::write`] (the `BlockMeta` without the anchor, which a
/// transmit stream does not use).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TxRecord {
    /// `BlockMeta::start` as passed to `write`.
    pub start: SampleIndex,
    /// `BlockMeta::flags` as passed to `write`.
    pub flags: BlockFlags,
    /// `BlockMeta::tx_time_ns` as passed to `write`.
    pub tx_time_ns: Option<i64>,
    /// Number of samples that were accepted.
    pub samples: usize,
    /// Index of the first accepted sample in the written stream (counted from 0).
    pub position: u64,
}

/// Virtual transmitter implementing [`TxStream`].
///
/// - `write` copies the accepted bytes into the caller's storage (a full storage accepts
///   fewer bytes, and none at all returns `BufferTooSmall`) and records the `BlockMeta` of
///   every call in a `heapless::Vec` of 64 entries (the 65th call fails with
///   `BufferTooSmall`).
/// - `poll_status` returns the injected statuses: `statuses` lists `(sample_index, flags)`
///   (sorted); a status is reported as soon as that many samples have been written, as a
///   block with `start = sample_index` and the given flags (`UNDERRUN`, `LATE`, ...).
///
/// ```
/// use sdrk_device::{formats, BlockFlags, BlockMeta, Rate, TxStream};
/// use sdrk_testkit::VirtualTx;
/// let mut storage = [0u8; 32];
/// let statuses = [(2u64, BlockFlags::UNDERRUN)];
/// let mut tx = VirtualTx::new(formats::CS8, Rate::hz(48_000)?, &mut storage, &statuses)?;
/// assert_eq!(tx.write(&[1, 2, 3, 4], &BlockMeta::default())?, 4);
/// let status = tx.poll_status()?.expect("an underrun at sample 2");
/// assert!(status.flags.contains(BlockFlags::UNDERRUN));
/// assert_eq!(tx.written(), &[1, 2, 3, 4]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug)]
pub struct VirtualTx<'a> {
    format: SampleFormat,
    rate: Rate,
    storage: &'a mut [u8],
    used: usize,
    records: heapless::Vec<TxRecord, MAX_RECORDS>,
    statuses: &'a [(u64, BlockFlags)],
    status_cursor: usize,
    samples: u64,
    granule_bytes: usize,
}

impl<'a> VirtualTx<'a> {
    /// Creates a virtual transmitter that stores data in `storage`.
    ///
    /// # Errors
    /// `InvalidParameter` if the format has no whole packing group or `statuses` is not
    /// sorted by index.
    #[allow(
        clippy::large_stack_frames,
        reason = "the 64 records are 3 KiB inside the value; clippy adds the Result and the temporaries on top, the value itself is built in the caller's slot"
    )]
    pub fn new(
        format: SampleFormat,
        rate: Rate,
        storage: &'a mut [u8],
        statuses: &'a [(u64, BlockFlags)],
    ) -> Result<Self, Error> {
        let granule_bytes = [1usize, 2, 4, 8]
            .into_iter()
            .find_map(|g| format.bytes_for_samples(g).ok())
            .ok_or(Error::InvalidParameter("no whole packing group"))?;
        if statuses
            .iter()
            .zip(statuses.iter().skip(1))
            .any(|(a, b)| a.0 > b.0)
        {
            return Err(Error::InvalidParameter("statuses must be sorted by index"));
        }
        Ok(Self {
            format,
            rate,
            storage,
            used: 0,
            records: heapless::Vec::new(),
            statuses,
            status_cursor: 0,
            samples: 0,
            granule_bytes,
        })
    }

    /// The bytes written so far.
    pub fn written(&self) -> &[u8] {
        self.storage.get(..self.used).unwrap_or(&[])
    }

    /// The recorded writes, oldest first.
    pub fn records(&self) -> &[TxRecord] {
        &self.records
    }

    /// Number of samples accepted so far.
    pub fn samples_written(&self) -> u64 {
        self.samples
    }
}

impl TxStream for VirtualTx<'_> {
    type Error = Error;

    fn format(&self) -> SampleFormat {
        self.format
    }

    fn sample_rate(&self) -> Rate {
        self.rate
    }

    fn write(&mut self, buf: &[u8], meta: &BlockMeta) -> Result<usize, Error> {
        if !buf.len().is_multiple_of(self.granule_bytes) {
            return Err(Error::InvalidParameter(
                "the data is not a whole number of samples",
            ));
        }
        if self.records.is_full() {
            return Err(Error::BufferTooSmall {
                needed: MAX_RECORDS + 1,
                got: MAX_RECORDS,
            });
        }
        let free = self.storage.len() - self.used;
        let take = buf.len().min(free);
        let take = take - take % self.granule_bytes;
        if take == 0 && !buf.is_empty() {
            return Err(Error::BufferTooSmall {
                needed: buf.len(),
                got: free,
            });
        }
        let src = buf.get(..take).ok_or(Error::Internal("source slice"))?;
        let dst = self
            .storage
            .get_mut(self.used..self.used + take)
            .ok_or(Error::Internal("storage slice"))?;
        dst.copy_from_slice(src);
        self.used += take;
        let samples = self.format.samples_in_bytes(take);
        // The capacity check above makes this push infallible.
        let _ = self.records.push(TxRecord {
            start: meta.start,
            flags: meta.flags,
            tx_time_ns: meta.tx_time_ns,
            samples,
            position: self.samples,
        });
        self.samples += samples as u64;
        Ok(take)
    }

    fn poll_status(&mut self) -> Result<Option<BlockMeta>, Error> {
        match self.statuses.get(self.status_cursor) {
            Some(&(at, flags)) if at <= self.samples => {
                self.status_cursor += 1;
                Ok(Some(BlockMeta {
                    start: SampleIndex(at),
                    flags,
                    anchor: None,
                    tx_time_ns: None,
                }))
            }
            _ => Ok(None),
        }
    }
}
