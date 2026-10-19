//! Raw sample files: bytes without a header.

use std::io::{ErrorKind as IoErrorKind, Read, Write};

use sdrk_device::{
    BlockFlags, BlockMeta, Rate, RxStream, SampleFormat, SampleIndex, TimeAnchor, TxStream,
};

use crate::error::{Error, Result};

/// Smallest number of samples that fills whole packing groups, and its size in bytes.
pub(crate) fn granule(format: &SampleFormat) -> Result<(usize, usize)> {
    for samples in [1usize, 2, 4, 8] {
        if let Ok(bytes) = format.bytes_for_samples(samples) {
            return Ok((samples, bytes));
        }
    }
    Err(Error::Unsupported(
        "sample format without a whole packing group",
    ))
}

/// Reads until `buf` is full or the reader is at its end; returns the bytes read.
pub(crate) fn read_full<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        let Some(rest) = buf.get_mut(filled..) else {
            break;
        };
        match reader.read(rest) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == IoErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(filled)
}

/// Reads raw samples from any reader.
///
/// Every `read` returns at most `chunk_samples` samples and only whole samples (whole packing
/// groups); a partial sample at the end of the file is dropped. At the end of the file `read`
/// returns zero bytes. The first block carries the anchor given to
/// [`RawReader::with_anchor`], if any.
///
/// ```
/// use sdrk_device::{formats, Rate, RxStream, SampleIndex};
/// use sdrk_io::RawReader;
/// let data = [1u8, 2, 3, 4, 5, 6];
/// let mut r = RawReader::new(&data[..], formats::CU8, Rate::hz(1000)?, SampleIndex(100), 2)?;
/// let mut buf = [0u8; 16];
/// let (n, meta) = r.read(&mut buf)?;
/// assert_eq!((n, meta.start.0), (4, 100)); // two samples of two bytes
/// let (n, meta) = r.read(&mut buf)?;
/// assert_eq!((n, meta.start.0), (2, 102));
/// assert_eq!(r.read(&mut buf)?.0, 0);
/// # Ok::<(), sdrk_io::Error>(())
/// ```
#[derive(Debug)]
pub struct RawReader<R: Read> {
    inner: R,
    format: SampleFormat,
    rate: Rate,
    next: u64,
    chunk_bytes: usize,
    granule_bytes: usize,
    anchor: Option<TimeAnchor>,
    eof: bool,
}

impl<R: Read> RawReader<R> {
    /// Creates a reader. `start` is the index of the first sample, `chunk_samples` the largest
    /// number of samples per `read`.
    ///
    /// # Errors
    /// `InvalidArgument` if `chunk_samples` is smaller than one packing group.
    pub fn new(
        inner: R,
        format: SampleFormat,
        rate: Rate,
        start: SampleIndex,
        chunk_samples: usize,
    ) -> Result<Self> {
        let (g_samples, g_bytes) = granule(&format)?;
        if chunk_samples < g_samples {
            return Err(Error::invalid(
                "chunk size is smaller than one packing group",
            ));
        }
        let chunk_bytes = format.bytes_for_samples(chunk_samples - chunk_samples % g_samples)?;
        Ok(Self {
            inner,
            format,
            rate,
            next: start.0,
            chunk_bytes,
            granule_bytes: g_bytes,
            anchor: None,
            eof: false,
        })
    }

    /// Attaches a time anchor to the first block.
    #[must_use]
    pub fn with_anchor(mut self, anchor: TimeAnchor) -> Self {
        self.anchor = Some(anchor);
        self
    }

    /// Index of the next sample that `read` will deliver.
    pub fn next_index(&self) -> SampleIndex {
        SampleIndex(self.next)
    }

    /// Gives back the underlying reader.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: Read> RxStream for RawReader<R> {
    type Error = Error;

    fn format(&self) -> SampleFormat {
        self.format
    }

    fn sample_rate(&self) -> Rate {
        self.rate
    }

    fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta)> {
        let want = buf.len().min(self.chunk_bytes);
        let want = want - want % self.granule_bytes;
        if want == 0 {
            return Err(Error::Device(sdrk_device::Error::BufferTooSmall {
                needed: self.granule_bytes,
                got: buf.len(),
            }));
        }
        let mut got = 0;
        if !self.eof {
            let target = buf
                .get_mut(..want)
                .ok_or_else(|| Error::invalid("buffer shorter than requested"))?;
            got = read_full(&mut self.inner, target)?;
            if got < want {
                self.eof = true;
            }
            got -= got % self.granule_bytes;
        }
        let samples = self.format.samples_in_bytes(got) as u64;
        let meta = BlockMeta {
            start: SampleIndex(self.next),
            flags: BlockFlags::empty(),
            anchor: if got > 0 { self.anchor.take() } else { None },
            tx_time_ns: None,
        };
        self.next = self.next.saturating_add(samples);
        Ok((got, meta))
    }
}

/// How often each flag was seen by a writer.
///
/// ```
/// use sdrk_io::FlagCounts;
/// assert_eq!(FlagCounts::default().gap_before, 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FlagCounts {
    /// Blocks with `GAP_BEFORE`.
    pub gap_before: u64,
    /// Blocks with `OVERFLOW`.
    pub overflow: u64,
    /// Blocks with `UNDERRUN`.
    pub underrun: u64,
    /// Blocks with `BURST_START`.
    pub burst_start: u64,
    /// Blocks with `BURST_END`.
    pub burst_end: u64,
    /// Blocks with `LATE`.
    pub late: u64,
    /// Blocks with `RETUNED`.
    pub retuned: u64,
}

impl FlagCounts {
    pub(crate) fn record(&mut self, flags: BlockFlags) {
        let pairs = [
            (BlockFlags::GAP_BEFORE, &mut self.gap_before),
            (BlockFlags::OVERFLOW, &mut self.overflow),
            (BlockFlags::UNDERRUN, &mut self.underrun),
            (BlockFlags::BURST_START, &mut self.burst_start),
            (BlockFlags::BURST_END, &mut self.burst_end),
            (BlockFlags::LATE, &mut self.late),
            (BlockFlags::RETUNED, &mut self.retuned),
        ];
        for (flag, counter) in pairs {
            if flags.contains(flag) {
                *counter += 1;
            }
        }
    }
}

/// Writes raw samples to any writer. Timing is ignored; the flags of the blocks are counted.
///
/// ```
/// use sdrk_device::{formats, BlockFlags, BlockMeta, Rate, TxStream};
/// use sdrk_io::RawWriter;
/// let mut w = RawWriter::new(Vec::new(), formats::CS8, Rate::hz(1000)?);
/// let meta = BlockMeta { flags: BlockFlags::BURST_START, ..BlockMeta::default() };
/// assert_eq!(w.write(&[1, 2, 3, 4], &meta)?, 4);
/// assert_eq!(w.flag_counts().burst_start, 1);
/// assert_eq!(w.into_inner(), vec![1, 2, 3, 4]);
/// # Ok::<(), sdrk_io::Error>(())
/// ```
#[derive(Debug)]
pub struct RawWriter<W: Write> {
    inner: W,
    format: SampleFormat,
    rate: Rate,
    counts: FlagCounts,
    samples: u64,
}

impl<W: Write> RawWriter<W> {
    /// Creates a writer for `format` at `rate`.
    pub fn new(inner: W, format: SampleFormat, rate: Rate) -> Self {
        Self {
            inner,
            format,
            rate,
            counts: FlagCounts::default(),
            samples: 0,
        }
    }

    /// The flags seen so far, counted per flag.
    pub fn flag_counts(&self) -> FlagCounts {
        self.counts
    }

    /// Number of samples written.
    pub fn samples_written(&self) -> u64 {
        self.samples
    }

    /// Flushes and gives back the underlying writer.
    pub fn into_inner(mut self) -> W {
        let _ = self.inner.flush();
        self.inner
    }
}

impl<W: Write> TxStream for RawWriter<W> {
    type Error = Error;

    fn format(&self) -> SampleFormat {
        self.format
    }

    fn sample_rate(&self) -> Rate {
        self.rate
    }

    fn write(&mut self, buf: &[u8], meta: &BlockMeta) -> Result<usize> {
        let samples = check_whole(&self.format, buf.len())?;
        self.inner.write_all(buf)?;
        self.counts.record(meta.flags);
        self.samples = self.samples.saturating_add(samples as u64);
        Ok(buf.len())
    }

    fn poll_status(&mut self) -> Result<Option<BlockMeta>> {
        Ok(None)
    }
}

/// Number of samples in `bytes`; an error unless `bytes` is a whole number of packing groups.
pub(crate) fn check_whole(format: &SampleFormat, bytes: usize) -> Result<usize> {
    let samples = format.samples_in_bytes(bytes);
    if format.bytes_for_samples(samples)? != bytes {
        return Err(Error::invalid(
            "the data is not a whole number of samples (packing groups)",
        ));
    }
    Ok(samples)
}
